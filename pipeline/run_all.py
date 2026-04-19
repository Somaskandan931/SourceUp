from pipeline.validate_merge import validate_and_merge
from pipeline.clean_normalize import clean
from pipeline.incremental_faiss import incremental_update

def main():
    validate_and_merge()
    clean()
    incremental_update()
    print("🚀 Full pipeline executed successfully")

if __name__ == "__main__":
    main()
