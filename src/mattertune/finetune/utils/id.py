import random
import string

def generate_random_id(k: int = 6) -> str:
    """
    Generate a random id with letters and digits
    Args:
        k: number of characters
    Returns:
        random_id: a random id
    """
    # letters + digits
    characters = string.ascii_letters + string.digits
    # random 6 characters
    random_id = ''.join(random.choices(characters, k=6))
    return random_id

# # 示例用法
# print(generate_random_id())
