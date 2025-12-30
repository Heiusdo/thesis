from datasets import load_dataset

ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")

# Kiểm tra thông tin dataset
print(ds)

# lưu data vào folder
ds["train"].to_csv("data/multiclass_sentiment_analysis_dataset_train.csv", index=False)
ds["validation"].to_csv("data/multiclass_sentiment_analysis_dataset_validation.csv", index=False)
ds["test"].to_csv("data/multiclass_sentiment_analysis_dataset_test.csv", index=False)
