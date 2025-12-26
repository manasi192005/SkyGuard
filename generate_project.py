import os

# Create directory structure
dirs = [
    'models', 'api', 'frontend', 'data/database', 
    'data/logs', 'data/suspects', 'data/uploads', 'data/outputs'
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created: {d}")

print("\nDirectory structure created!")
print("\nNext: I'll provide the code files one by one.")
