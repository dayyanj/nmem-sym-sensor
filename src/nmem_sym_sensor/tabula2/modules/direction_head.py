# TABULA2/modules/direction_head.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonauralDirectionHead(nn.Module):
    """
    Monaural direction prediction head using spectral and temporal cues.
    
    Predicts azimuth (horizontal angle) from -180° to +180° using:
    - Spectral shape analysis (HRTF-like filtering effects)
    - Temporal envelope patterns
    - Distance estimation via intensity/spectral rolloff
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_azimuth_bins: int = 72,  # 5° resolution
        use_distance: bool = True,
        use_elevation: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_azimuth_bins = num_azimuth_bins
        self.use_distance = use_distance
        self.use_elevation = use_elevation
        self.azimuth_resolution = 360.0 / num_azimuth_bins

        # Multi-scale feature extraction for direction cues
        self.spectral_analyzer = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(16, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, dilation=2),
            nn.ReLU(),
            nn.GroupNorm(16, hidden_dim),
        )

        # Temporal pattern analysis for direction
        self.temporal_analyzer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.GroupNorm(16, hidden_dim),
            nn.AdaptiveAvgPool1d(1)  # Global temporal pooling
        )

        # Direction prediction heads
        self.azimuth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_azimuth_bins)
        )

        # Optional distance estimation (near/far classification + regression)
        if use_distance:
            self.distance_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3)  # near/medium/far
            )
            self.distance_regressor = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # log-distance estimate
            )

        # Optional elevation (future extension)
        if use_elevation:
            self.elevation_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 36)  # -90° to +90°, 5° resolution
            )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: [B, D, T] - voice or noise features from disentangler
            
        Returns:
            Dict containing:
            - azimuth_logits: [B, num_bins] - azimuth angle probabilities
            - azimuth_degrees: [B] - predicted azimuth in degrees
            - confidence: [B] - prediction confidence [0,1]
            - distance_class: [B, 3] - near/medium/far (if enabled)
            - distance_meters: [B] - estimated distance (if enabled)
        """
        B, D, T = features.shape

        # Extract spectral and temporal direction cues
        spectral_features = self.spectral_analyzer(features)  # [B, H, T]
        temporal_features = self.temporal_analyzer(spectral_features).squeeze(-1)  # [B, H]

        # Azimuth prediction
        azimuth_logits = self.azimuth_head(temporal_features)  # [B, num_bins]
        azimuth_probs = F.softmax(azimuth_logits, dim=-1)

        # Convert to degrees (circular regression)
        bin_centers = torch.linspace(-180, 180 - self.azimuth_resolution,
                                   self.num_azimuth_bins, device=features.device)
        azimuth_degrees = self._circular_expectation(azimuth_probs, bin_centers)

        # Confidence based on entropy
        confidence = 1.0 - (-azimuth_probs * torch.log(azimuth_probs + 1e-8)).sum(dim=-1) / math.log(self.num_azimuth_bins)

        results = {
            'azimuth_logits': azimuth_logits,
            'azimuth_degrees': azimuth_degrees,
            'azimuth_probs': azimuth_probs,
            'confidence': confidence
        }

        # Optional distance estimation
        if self.use_distance:
            distance_class_logits = self.distance_classifier(temporal_features)
            distance_log = self.distance_regressor(temporal_features).squeeze(-1)

            results.update({
                'distance_class_logits': distance_class_logits,
                'distance_class_probs': F.softmax(distance_class_logits, dim=-1),
                'distance_meters': torch.exp(distance_log)  # Convert from log-space
            })

        # Optional elevation
        if self.use_elevation:
            elevation_logits = self.elevation_head(temporal_features)
            elevation_probs = F.softmax(elevation_logits, dim=-1)
            elevation_bins = torch.linspace(-90, 90, 36, device=features.device)
            elevation_degrees = (elevation_probs * elevation_bins).sum(dim=-1)

            results.update({
                'elevation_logits': elevation_logits,
                'elevation_degrees': elevation_degrees,
                'elevation_probs': elevation_probs
            })

        return results

    def _circular_expectation(self, probs: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
        """Compute circular mean for angle prediction"""
        # Convert to radians for circular math
        angles_rad = bin_centers * math.pi / 180.0

        # Circular expectation using complex exponentials
        cos_sum = (probs * torch.cos(angles_rad)).sum(dim=-1)
        sin_sum = (probs * torch.sin(angles_rad)).sum(dim=-1)

        # Convert back to degrees
        angle_rad = torch.atan2(sin_sum, cos_sum)
        angle_deg = angle_rad * 180.0 / math.pi

        return angle_deg


class DirectionLoss(nn.Module):
    """Loss function for direction prediction with circular angle handling"""

    def __init__(self,
                 azimuth_weight: float = 1.0,
                 distance_weight: float = 0.5,
                 confidence_weight: float = 0.1):
        super().__init__()
        self.azimuth_weight = azimuth_weight
        self.distance_weight = distance_weight
        self.confidence_weight = confidence_weight

    def forward(self,
                predictions: dict[str, torch.Tensor],
                targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: Output from MonauralDirectionHead
            targets: Dict with keys:
                - azimuth_degrees: [B] ground truth azimuth
                - azimuth_bins: [B] ground truth bin indices (optional)
                - distance_class: [B] distance class labels (optional)
                - distance_meters: [B] distance in meters (optional)
        """
        losses = {}
        total_loss = 0.0

        # Azimuth loss (circular)
        if 'azimuth_bins' in targets:
            # Classification loss
            azimuth_loss = F.cross_entropy(predictions['azimuth_logits'], targets['azimuth_bins'])
        else:
            # Circular regression loss
            pred_deg = predictions['azimuth_degrees']
            true_deg = targets['azimuth_degrees']
            azimuth_loss = self._circular_l1_loss(pred_deg, true_deg)

        losses['azimuth'] = azimuth_loss
        total_loss += self.azimuth_weight * azimuth_loss

        # Distance losses
        if 'distance_class_logits' in predictions and 'distance_class' in targets:
            dist_class_loss = F.cross_entropy(predictions['distance_class_logits'], targets['distance_class'])
            losses['distance_class'] = dist_class_loss
            total_loss += self.distance_weight * dist_class_loss

        if 'distance_meters' in predictions and 'distance_meters' in targets:
            # Log-space MSE for distance regression
            pred_log_dist = torch.log(predictions['distance_meters'] + 1e-6)
            true_log_dist = torch.log(targets['distance_meters'] + 1e-6)
            dist_reg_loss = F.mse_loss(pred_log_dist, true_log_dist)
            losses['distance_regression'] = dist_reg_loss
            total_loss += self.distance_weight * dist_reg_loss

        # Confidence regularization (encourage high confidence for good predictions)
        if 'confidence' in predictions:
            # Higher confidence should correlate with lower azimuth error
            azimuth_error = torch.abs(self._circular_distance(
                predictions['azimuth_degrees'], targets['azimuth_degrees']
            ))
            confidence_loss = F.mse_loss(
                predictions['confidence'],
                1.0 - azimuth_error / 180.0  # Normalize error to [0,1]
            )
            losses['confidence'] = confidence_loss
            total_loss += self.confidence_weight * confidence_loss

        losses['total'] = total_loss
        return losses

    def _circular_l1_loss(self, pred_degrees: torch.Tensor, true_degrees: torch.Tensor) -> torch.Tensor:
        """L1 loss accounting for circular nature of angles"""
        diff = self._circular_distance(pred_degrees, true_degrees)
        return diff.abs().mean()

    def _circular_distance(self, angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
        """Compute shortest angular distance between two angles"""
        diff = angle1 - angle2
        # Wrap to [-180, 180]
        diff = ((diff + 180) % 360) - 180
        return diff


# Integration utility for existing DisentanglerModel
class DisentanglerWithDirection(nn.Module):
    """Wrapper that adds direction prediction to existing DisentanglerModel"""

    def __init__(self,
                 disentangler_model: nn.Module,
                 direction_config: dict | None = None):
        super().__init__()

        self.disentangler = disentangler_model

        # Default direction head config
        default_config = {
            "input_dim": 128,
            "hidden_dim": 256,
            "num_azimuth_bins": 72,
            "use_distance": True,
            "use_elevation": False
        }

        config = {**default_config, **(direction_config or {})}

        # Direction heads for both voice and noise pathways
        self.voice_direction_head = MonauralDirectionHead(**config)
        self.noise_direction_head = MonauralDirectionHead(**config)

    def forward(self, x: torch.Tensor, predict_direction: bool = True) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Input audio [B, 1, T]
            predict_direction: Whether to run direction prediction
            
        Returns:
            All disentangler outputs plus direction predictions
        """
        # Get base disentangler outputs
        outputs = self.disentangler(x)

        if predict_direction:
            # Predict direction from voice and noise pathways
            voice_direction = self.voice_direction_head(outputs['voice_fused'])
            noise_direction = self.noise_direction_head(outputs['noise_fused'])

            # Add direction predictions to outputs
            outputs.update({
                'voice_direction': voice_direction,
                'noise_direction': noise_direction
            })

        return outputs
