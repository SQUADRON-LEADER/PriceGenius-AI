# Quick Start Guide - Smart Product Pricing Challenge

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```powershell
cd c:\Users\aayus\OneDrive\Desktop\AMAZON\68e8d1d70b66d_student_resource\student_resource
pip install -r requirements.txt
```

### Step 2: Run Training
```powershell
python src\train_complete.py
```

### Step 3: Submit
- Find your output file: `dataset/test_out.csv`
- Upload to the challenge portal
- Submit your documentation

---

## 📊 What the Model Does

### 1. **Text Feature Extraction** (Main Focus)
- Extracts numeric features (prices, quantities, weights)
- TF-IDF vectorization of product descriptions
- Domain-specific features (brand indicators, units)

### 2. **Machine Learning Models**
- **LightGBM:** Fast gradient boosting
- **XGBoost:** Robust gradient boosting
- **Ensemble:** Averages both models for better predictions

### 3. **Training Process**
- 5-fold cross-validation
- Log transformation for price prediction
- Early stopping to prevent overfitting

---

## 📁 File Structure

```
student_resource/
├── dataset/
│   ├── train.csv              # Training data (75K products)
│   ├── test.csv               # Test data (75K products)
│   └── test_out.csv           # YOUR PREDICTIONS (generated)
├── src/
│   ├── train_complete.py      # 🔥 MAIN SCRIPT - Run this!
│   ├── config.py              # Configuration
│   ├── feature_engineering.py # Feature extraction
│   └── utils.py               # Utilities
├── requirements.txt           # Dependencies
└── SOLUTION_DOCUMENTATION.md  # Full documentation
```

---

## ⚙️ Configuration Options

In `src/train_complete.py`, you can modify:

### Basic Settings (in config.py):
```python
RANDOM_SEED = 42          # For reproducibility
N_FOLDS = 5               # Cross-validation folds
```

### Advanced Settings:
```python
USE_IMAGES = False        # Set to True to use image features
                          # (requires TensorFlow installation)
```

---

## 🎯 Expected Performance

### Text-Only Model (Default):
- **Training Time:** 10-20 minutes (on standard CPU)
- **Expected SMAPE:** 15-25%
- **Memory Usage:** ~4-6 GB

### With Image Features (Optional):
- **Training Time:** 2-4 hours (depends on image download/processing)
- **Expected SMAPE:** 13-22% (2-5% improvement)
- **Memory Usage:** ~8-12 GB
- **Requirements:** TensorFlow, GPU recommended

---

## 🔧 Troubleshooting

### Issue: "Module not found"
**Solution:** Install dependencies
```powershell
pip install numpy pandas scikit-learn lightgbm xgboost
```

### Issue: "File too large" or slow loading
**Solution:** This is normal - train.csv and test.csv are large files

### Issue: Out of memory
**Solution:** Reduce batch size or disable image features

### Issue: TensorFlow errors
**Solution:** Image features are optional. Keep `USE_IMAGES = False`

---

## 📈 Model Improvement Tips

### Quick Wins:
1. **Feature Engineering:**
   - Extract brand names more precisely
   - Add more domain-specific keywords
   - Parse item pack quantity better

2. **Hyperparameter Tuning:**
   - Adjust `learning_rate` (try 0.03, 0.05, 0.1)
   - Modify `num_leaves` in LightGBM
   - Change `max_depth` in XGBoost

3. **Ensemble Weights:**
   - Instead of simple average, try weighted average
   - Weight models by validation performance

### Advanced Improvements:
1. **Better Text Embeddings:**
   - Use sentence-transformers for semantic similarity
   - Try pre-trained BERT models (check parameter limits)

2. **Image Features:**
   - Download all images (takes time)
   - Use different vision models (EfficientNet, ResNet)

3. **Additional Models:**
   - Add CatBoost to ensemble
   - Try neural networks for specific product categories

---

## 📝 Documentation Checklist

Before submission, ensure:
- ✅ `test_out.csv` has exactly 75,000 rows
- ✅ All `sample_id` values match `test.csv`
- ✅ All prices are positive floats
- ✅ Documentation describes your approach
- ✅ No external price lookup was used

---

## 🎓 Understanding SMAPE

**SMAPE = Symmetric Mean Absolute Percentage Error**

```
SMAPE = (1/n) × Σ |predicted - actual| / ((|actual| + |predicted|)/2) × 100
```

**Example:**
- Actual price: $100
- Predicted price: $120
- SMAPE = |100-120| / ((100+120)/2) × 100 = 18.18%

**Lower is better!** Target: <20%

---

## 🚀 Next Steps After Training

1. **Check Results:**
   ```powershell
   head dataset\test_out.csv
   ```

2. **Validate Output:**
   - Ensure 75,000 predictions
   - Check for negative/zero prices
   - Compare with sample_test_out.csv format

3. **Iterate:**
   - Analyze errors on validation set
   - Try different features
   - Tune hyperparameters

4. **Submit:**
   - Upload `test_out.csv` to portal
   - Submit `SOLUTION_DOCUMENTATION.md`

---

## 💡 Pro Tips

1. **Start Simple:** Run the default model first, then add complexity
2. **Monitor Progress:** Watch the cross-validation scores
3. **Save Models:** Models are saved to `models/pricing_model.pkl`
4. **Feature Importance:** Check which features matter most
5. **Time Management:** Text-only model is fast and effective

---

## 📚 Additional Resources

- **LightGBM Docs:** https://lightgbm.readthedocs.io/
- **XGBoost Docs:** https://xgboost.readthedocs.io/
- **Scikit-learn:** https://scikit-learn.org/

---

## ✅ Compliance Checklist

- ✅ No external price lookup
- ✅ Only provided training data used
- ✅ All models have appropriate licenses (MIT/Apache 2.0)
- ✅ Model parameters < 8 Billion
- ✅ Output format matches requirements

---

**Good luck with your challenge! 🎯**

For questions, review the full documentation in `SOLUTION_DOCUMENTATION.md`
