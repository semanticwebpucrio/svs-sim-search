# TODO: load values from .env file
# constants
REDIS_HOST = "redis"
REDIS_PORT = 6379

API_PORT = 8080
API_HOST = "0.0.0.0"
API_DESCRIPTION = """
## Multimodal Clustering using Product Quantization 🚀
"""

# single instances
api_app = None
redis_cli = None
sub_pre_text = None
sub_text = None
sub_pre_image = None
sub_image = None
