from .sinhala_character_predictor import SinhalaCharacterPredictor

class MainPredictor:
    def __init__(self, user_need="char"):
        valid_needs = ["char", "era", "both"]
        if user_need not in valid_needs:
            raise ValueError(f"Invalid option for user_need. Choose from {valid_needs}")
        self.user_need = user_need
        self.char_predictor = SinhalaCharacterPredictor()

    def predict(self, img_path):
        return self.char_predictor.predict(img_path)
"""from sinhala_character_predictor import SinhalaCharacterPredictor

class MainPredictor:
    def __init__(self, user_need="char"):
       
        valid_needs = ["char", "era", "both"]
        if user_need not in valid_needs:
            raise ValueError(f"Invalid option for user_need. Choose from {valid_needs}")
        self.user_need = user_need
        self.char_predictor = SinhalaCharacterPredictor()

    def predict(self, img_path):
       
        if self.user_need in ["char", "both"]:
            # SinhalaCharacterPredictor.predict returns a dict with keys:
            # "Random Forest", "Extra Trees", "XGBoost", "Final"
            return self.char_predictor.predict(img_path)
        else:
            return {"Final": "N/A"}"""
