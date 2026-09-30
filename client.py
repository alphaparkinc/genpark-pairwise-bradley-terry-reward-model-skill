"""Bradley-Terry Reward Model.
100% Python Standard Library.
"""

import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

class BradleyTerryRewardModel:
    """Bradley-Terry preference model computing choice probabilities and preference losses."""
    @staticmethod
    def predict_preference_prob(r_chosen: float, r_rejected: float) -> float:
        return round(sigmoid(r_chosen - r_rejected), 4)

    @staticmethod
    def compute_loss(r_chosen: float, r_rejected: float) -> float:
        prob = sigmoid(r_chosen - r_rejected)
        loss = -math.log(max(1e-12, prob))
        return round(loss, 4)

    @staticmethod
    def gradient_step(r_chosen: float, r_rejected: float, lr: float = 0.05) -> dict:
        prob = sigmoid(r_chosen - r_rejected)
        grad = 1.0 - prob
        new_chosen = round(r_chosen + lr * grad, 4)
        new_rejected = round(r_rejected - lr * grad, 4)
        return {
            "grad": round(grad, 4),
            "updated_r_chosen": new_chosen,
            "updated_r_rejected": new_rejected,
            "new_loss": BradleyTerryRewardModel.compute_loss(new_chosen, new_rejected)
        }
