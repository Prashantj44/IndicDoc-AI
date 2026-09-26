# Dataset Documentation

## IndicDLP (Indic Document Layout Parsing)

### Overview

| Property | Details |
|---|---|
| **Name** | IndicDLP |
| **Source** | [AIKosh](https://aikosh.indiaai.gov.in/home) / [HuggingFace](https://huggingface.co/datasets/IndicDLP/IndicDLP-dataset) |
| **Origin** | AI4Bharat / IIT research collaboration |
| **Task** | Document Layout Parsing (Object Detection) |
| **Total Images** | 119,806 |
| **Annotation Format** | COCO-style JSON |
| **Number of Classes** | 42 physical and logical layout classes |

### Languages (12)

| # | Language | Script |
|---|----------|--------|
| 1 | Assamese | Assamese (Eastern Nagari) |
| 2 | Bengali | Bengali |
| 3 | English | Latin |
| 4 | Gujarati | Gujarati |
| 5 | Hindi | Devanagari |
| 6 | Kannada | Kannada |
| 7 | Malayalam | Malayalam |
| 8 | Marathi | Devanagari |
| 9 | Odia | Odia |
| 10 | Punjabi | Gurmukhi |
| 11 | Tamil | Tamil |
| 12 | Telugu | Telugu |

### Document Domains (12)

1. Novels
2. Textbooks
3. Magazines
4. Acts & Rules
5. Research Papers
6. Manuals
7. Brochures
8. Syllabi
9. Question Papers
10. Notices
11. Forms
12. Newspapers

### Classes (42)

The dataset annotates 42 physical and logical document layout classes:

Paragraph, Image, Table, Header, Footer, Title, Caption, Page Number, Footnote, List, Figure, Equation, Logo, Stamp, Signature, Handwriting, Chart, Map, Separator, Advertisement, Watermark, Background, Margin Note, Column, Section Header, Sub Header, Abstract, Author, Affiliation, Date, Reference, Acknowledgment, Appendix, Table of Contents, Index, Glossary, Bibliography, Preface, Dedication, Colophon, Running Header, Running Footer

### Annotation Format

Annotations follow the COCO Object Detection format:

```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "area": ...,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "Paragraph", "supercategory": "document"}
  ]
}
```

### Dataset Access

- **AIKosh**: [aikosh.indiaai.gov.in](https://aikosh.indiaai.gov.in/home)
- **HuggingFace**: [IndicDLP/IndicDLP-dataset](https://huggingface.co/datasets/IndicDLP/IndicDLP-dataset)

### License

Please refer to the [HuggingFace dataset page](https://huggingface.co/datasets/IndicDLP/IndicDLP-dataset) for the specific license terms.

### Suitability

This dataset is excellent for:
- Document layout detection/parsing with deep learning
- Multilingual document understanding research
- Transfer learning from COCO-pretrained object detection models
- Evaluation across diverse Indian scripts and document types

### Sample Data

For development and testing, we provide a synthetic sample dataset generator:

```bash
python -m ml.datasets.create_sample_data
```

This creates 30 synthetic document images with random bounding box annotations in `data/processed/sample/`. This is strictly for pipeline testing and not for model performance evaluation.
