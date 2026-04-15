from keras.models import load_model

old_path = r"C:\Users\HP\OneDrive\Desktop\Stock Price\stock_price_prediction\stock_dl_model.h5"
new_path = r"C:\Users\HP\OneDrive\Desktop\Stock Price\stock_price_prediction\stock_dl_model.keras"

print("Loading old .h5 model...")
model = load_model(old_path, compile=False, safe_mode=False)

print("Saving model in new .keras format...")
model.save(new_path)

print(f"✅ Done! New model saved at: {new_path}")
