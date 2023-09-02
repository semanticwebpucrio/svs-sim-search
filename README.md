# Staged Vector Stream Similarity Search Methods

### Authors:
- João Pedro V. Pinheiro [jpinheiro@inf.puc-rio.br](mailto:jpinheiro@inf.puc-rio.br)
- Lucas R. Borges [lborges@inf.puc-rio.br](mailto:lborges@inf.puc-rio.br)
- Bruno F. Martins da Silva [bsilva@inf.puc-rio.br](mailto:bsilva@inf.puc-rio.br)
- Luiz A. P. Paes Leme [lapaesleme@ic.uff.br](mailto:lapaesleme@ic.uff.br)
- Marco A. Casanova [casanova@inf.puc-rio.br](mailto:casanova@inf.puc-rio.br)

### Introduction:
>This article describes a family of methods, called staged vector stream similarity search methods, or briefly SVS, to help address the vector stream similarity search problem, defined as: “Given a (high-dimensional) vector q and a time interval T, find a ranked list of vectors, retrieved from a vector stream, that are similar to q and that were received in the time interval T”. The main feature of SVS lies in that it does not depend on having the full set of vectors available beforehand, but adapts to the vector stream as the vectors are received. The article describes experiments to assess the performance of two implementations of SVS, one based on product quantization, called staged IVFADC, and an other based on Hierarchical Navigable Small World graphs, called staged HNSW. To test SVS in practice, the article discusses a proof-of-concept implementation of a classified ad retrieval tool that uses staged HNSW.

#### Keywords:
- High-dimensional Vector Streams
- Approximate Nearest Neighbor Search
- Product Quantization
- Hierarchical Navigable Small World Graphs
- Classified Ad
