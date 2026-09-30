from client import BradleyTerryRewardModel

prob = BradleyTerryRewardModel.predict_preference_prob(2.5, 1.0)
print("Preference probability P(chosen > rejected):", prob)
step = BradleyTerryRewardModel.gradient_step(2.5, 1.0)
print("Updated rewards after step:", step)
