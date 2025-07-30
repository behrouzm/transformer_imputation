# Transformer Imputation for Methylation Data

⚠️ **EXPERIMENTAL CODE - EARLY DEVELOPMENT STAGE** ⚠️

This repository contains experimental transformer-based approaches for imputing missing methylation values in genomic data. **This code is for research purposes only and is not intended for production use.**

## ⚠️ Important Notice

- **This is experimental research code in early development**
- **Not tested for production use**
- **May contain bugs and incomplete features**
- **Use at your own risk**
- **No warranty or support provided**

## Overview

DNA methylation is a crucial epigenetic modification that regulates gene expression. Missing methylation values in datasets can significantly impact downstream analyses. This project implements two experimental transformer-based models for methylation imputation:

1. **TabTransformer** - A simpler transformer architecture
2. **FT-Transformer** - A more sophisticated transformer with CpG-specific embeddings

Both models use a leave-one-out cross-validation strategy for evaluation.

## Models

### TabTransformer (`tab_transformer.py`)
- **Architecture**: Simple transformer with feature embeddings
- **Status**: Experimental implementation
- **Key Features**: 
  - Feature embeddings for methylation values
  - Multi-head attention mechanism
  - Sigmoid activation for output

### FT-Transformer (`ft_transformer.py`)
- **Architecture**: Advanced transformer with multiple embedding types
- **Status**: Experimental implementation
- **Key Features**:
  - CpG site-specific embeddings
  - Value embeddings for methylation levels
  - Missing value embeddings for NaN handling
  - Position embeddings for sequence information
  - Multi-head attention with residual connections

## Experimental Nature

This code is part of ongoing research and experimentation:
- **Not optimized for performance**
- **May have incomplete documentation**
- **Architecture and parameters subject to change**
- **No guarantee of accuracy or reliability**
- **Intended for academic research only**

## License

**RESTRICTED USE LICENSE**

Copyright (c) 2025 Behrouz Mollashahi

**ALL RIGHTS RESERVED**

This software and associated documentation files (the "Software") are provided for **RESEARCH PURPOSES ONLY**. 

**NO PERMISSION IS GRANTED** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

**THE SOFTWARE IS PROVIDED "AS IS"**, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

**USE OF THIS SOFTWARE IS STRICTLY PROHIBITED** without explicit written permission from the copyright holder.

## Citation and References

If you reference this work in any academic publication, you must:

1. **Cite the original transformer papers**:
   - Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
   - Gorishniy, Y., et al. "Revisiting deep learning models for tabular data." Advances in Neural Information Processing Systems 34 (2021).

2. **Contact the author** for any questions or collaboration requests.

## Disclaimer

This code is provided for **ACADEMIC RESEARCH PURPOSES ONLY**. The authors make no representations about the suitability of this software for any purpose. The software is provided without warranty of any kind.

**DO NOT USE THIS CODE** for any commercial, clinical, or production applications without explicit written permission.

## Contact

For research collaboration or questions about this experimental work, please contact:
**Behrouz Mollashahi** - behrouzmolla@gmail.com

---

**⚠️ WARNING: This is experimental research code. Use at your own risk. ⚠️**