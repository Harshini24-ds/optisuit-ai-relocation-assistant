import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

libraries = [
    # Data handling
    "pandas",
    "numpy",
    "openpyxl",

    # Machine Learning
    "scikit-learn",
    "xgboost",
    "imbalanced-learn",

    # Visualization
    "matplotlib",
    "seaborn",
    "plotly",

    # Dashboard
    "streamlit",

    # Model saving
    "joblib",

    "flask",

    # OpenAI API
    "openai",

    "deep-translator"
]

print("=" * 50)
print("OPTISUIT - INSTALLING ALL REQUIRED LIBRARIES")
print("=" * 50)

for lib in libraries:
    print(f"\n📦 Installing: {lib} ...")
    try:
        install(lib)
        print(f"✅ {lib} installed successfully!")
    except Exception as e:
        print(f"❌ Failed to install {lib}: {e}")

print("\n" + "=" * 50)
print("✅ ALL LIBRARIES INSTALLED SUCCESSFULLY!")
print("=" * 50)