class Config:
    """simple class to store the config"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)