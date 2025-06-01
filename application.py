import torch
import json
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from docx import Document  
import os
import json
from pathlib import Path

# Configuration
MODEL_DIR = "Legal-BERT_uncased_output"  # Directory containing config.json and pytorch_model.bin
LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
MAX_LENGTH = 128  # Model max input length

class TextProcessor:
    """Text processing utilities"""

    @staticmethod
    def read_docx_file(file_path):
        """Read content from .docx file"""
        try:
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Failed to read file {file_path}: {str(e)}")

    @staticmethod
    def split_sentences(text):
        """English sentence segmentation"""
        if not text.strip():
            return []
            
        # Handle multiple newlines and whitespace
        text = re.sub(r'\n+', '\n', text.strip())
        # Split on sentence boundaries
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        # Filter out short sentences (less than 3 chars) and empty strings
        return [s.strip() for s in sentences if len(s.strip()) >= 3]

class TextClassifier:
    """Text classification model"""

    def __init__(self, model_dir):
        try:
            # Load model (automatically uses CPU)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                local_files_only=True
            )
            self.model.eval()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
        except Exception as e:
            raise Exception(f"Failed to load model from {model_dir}: {str(e)}")

    def predict(self, text):
        """Predict label for a single sentence"""
        if not text.strip():
            return {
                "sentence": text,
                "label": "UNKNOWN",
                "confidence": 0.0,
                "label_scores": {label: 0.0 for label in LABELS}
            }

        try:
            inputs = self.tokenizer(
                text,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_label = torch.argmax(probs, dim=-1).item()

            return {
                "sentence": text,
                "label": LABELS[pred_label],
                "confidence": round(probs[0][pred_label].item(), 4),
                "label_scores": {label: round(probs[0][i].item(), 4) for i, label in enumerate(LABELS)}
            }
        except Exception as e:
            print(f"Error processing sentence: {text[:100]}... Error: {str(e)}")
            return {
                "sentence": text,
                "label": "ERROR",
                "confidence": 0.0,
                "label_scores": {label: 0.0 for label in LABELS}
            }

def analyze_text_file(input_file, output_file="/home/zchen629/appication/output/CACC000074_2019-output.json"):
    """Process text file and save results"""
    # Verify input file exists
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read and preprocess text
    try:
        raw_text = TextProcessor.read_docx_file(input_file)  # 修改为读取.docx文件
        sentences = TextProcessor.split_sentences(raw_text)
    except Exception as e:
        raise Exception(f"Failed to process input text: {str(e)}")

    # Initialize classifier
    try:
        classifier = TextClassifier(MODEL_DIR)
    except Exception as e:
        raise Exception(f"Failed to initialize classifier: {str(e)}")

    # Process sentences
    results = []
    error_count = 0
    
    for i, sent in enumerate(sentences, 1):
        print(f"Processing sentence {i}/{len(sentences)}: {sent[:50]}...")
        result = classifier.predict(sent)
        results.append(result)
        if result["label"] in ["ERROR", "UNKNOWN"]:
            error_count += 1

    # Prepare output structure
    output = {
        "source_file": str(Path(input_file).name),
        "total_sentences": len(sentences),
        "processed_sentences": len(results) - error_count,
        "error_count": error_count,
        "sentences": results,
        "label_distribution": {
            label: sum(1 for item in results if item["label"] == label)
            for label in LABELS + ["ERROR", "UNKNOWN"]
        }
    }

    # Save results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Analysis complete. Results saved to {output_file}")
    except Exception as e:
        raise Exception(f"Failed to save results to {output_file}: {str(e)}")

    return output

if __name__ == "__main__":
    input_file = "/home/zchen629/appication/2019/CACC000074X_2019.docx"  # Input path
    output_dir = "/home/zchen629/appication/output/"  # Output directory

    # 2. 自动创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 3. 保存结果
    output_path = os.path.join(output_dir, "CACC000074X_2019-output.json")
    
    try:
        results = analyze_text_file(input_file, output_path)  # 使用 output_path 作为参数
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
