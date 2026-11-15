# References

- S. A. Mirsalari et al., “StreamEase: Enabling Real-Time Inference of Temporal Convolution Networks on Low-Power MCUs with Stream-Oriented Automatic Transformation,” IEEE ICECS, 2024. [DOI 10.1109/ICECS61496.2024.10848742](https://doi.org/10.1109/ICECS61496.2024.10848742)
- S. A. Mirsalari et al., “Enabling Real-Time Streaming Temporal Convolution Network Inference on Ultra-Low-Power Microcontrollers,” IEEE ISVLSI, 2025. [DOI 10.1109/ISVLSI65124.2025.11130291](https://doi.org/10.1109/ISVLSI65124.2025.11130291)
- L. Cavigelli, P. Degen, and L. Benini, “CBinfer: Change-Based Inference for Convolutional Neural Networks on Video Data,” ICDSC, 2017. [ETH Zürich Research Collection](https://www.research-collection.ethz.ch/entities/publication/0ca777b8-c8fd-4f16-8f2d-59235f3cf15a)
- M. Parger et al., “DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos,” CVPR, 2022. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html)
- G. Weiss, “WISDM Smartphone and Smartwatch Activity and Biometrics Dataset,” UCI Machine Learning Repository, 2019. [DOI 10.24432/C5HK59](https://doi.org/10.24432/C5HK59)
- O. Banos, R. Garcia, and A. Saez, “MHEALTH,” UCI Machine Learning Repository, 2014. [DOI 10.24432/C5TW22](https://doi.org/10.24432/C5TW22)
- Antmicro, “Execution metrics, profiling and opcode counting,” Renode documentation. [Renode docs](https://renode.readthedocs.io/en/latest/execution-tracing/metrics-and-profiling.html)

StreamEase provides context for efficient forward streaming. CBinfer and DeltaCNN provide related sparse change propagation in video. The experiment asks what selective score correction is worth under an explicit MCU history bound when samples arrive late. These references motivate the design and its baselines.
