class UserNotFoundException(Exception):
    """
    Excepción lanzada cuando no se encuentra un usuario en el feature store
    """
    

class PredictionException(Exception):
    """
    Excepción lanzada cuando ocurre un error durante la inferencia del modelo
    """