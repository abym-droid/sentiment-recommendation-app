import gzip
import pickle
import os

def decompress_models():
    """Decompress model files for Heroku deployment"""
    
    compressed_files = [
        ('models/best_sentiment_rf_model.pkl.gz', 'models/best_sentiment_rf_model.pkl'),
        ('models/tfidf_vectorizer.pkl.gz', 'models/tfidf_vectorizer.pkl'),
        ('models/user_final_rating.pkl.gz', 'models/user_final_rating.pkl'),
        ('data/df_nlp_prep.pkl.gz', 'data/df_nlp_prep.pkl')
    ]
    
    for compressed_file, output_file in compressed_files:
        if os.path.exists(compressed_file) and not os.path.exists(output_file):
            print(f"Decompressing {compressed_file}...")
            try:
                with gzip.open(compressed_file, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                print(f"Successfully created {output_file}")
            except Exception as e:
                print(f"Error decompressing {compressed_file}: {e}")
        else:
            if os.path.exists(output_file):
                print(f"{output_file} already exists, skipping...")
            else:
                print(f"Warning: {compressed_file} not found")

if __name__ == "__main__":
    decompress_models()