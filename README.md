# BiverWordle: Visualizing Stock Market Sentiment with Financial Text Data and Trends

## Overview

![Overview of visualization method BiverWordle, using example stock 002594 BYD](https://github.com/Brian-Lei-XIA/BiverWordle/blob/main/images/overview.png)

BiverWordle is an innovative visualization system specifically designed to analyze, visualize, and authenticate sentiments relating to stock trends within financial text forums. This system ingeniously merges a time-series financial data representation, which is inspired by a candlestick chart, with a stacked area chart/streamgraph derived from text classification results. It further integrates word clouds that are uniquely arranged in the shapes of sad, neutral, and happy smiley face ideograms. These combined elements provide a comprehensive and intuitive portrayal of stock market sentiment.

The name "BiverWordle" encapsulates the essence of our approach, alluding to two river-like area charts flowing with words. Our system's effectiveness is evaluated through various measures. These include text classification metrics (accuracy, precision, recall, and F1-score) for test data, and additionally, three use cases that vividly demonstrate the interactive data exploration possibilities of our visualization tool.

## News
The paper has been accepted at VINCI 2023 as a short paper!

## Key Features

- Integration of multiple visualization techniques for a comprehensive understanding of stock market sentiment.
- Analysis of Chinese financial text data to understand and visualize market trends.
- Validation of our system using three different stock trend scenarios.
- Exploration of potential improvements in both the financial text classification and visualization aspects of the project.

## System Components

- **Text Classification**: BiverWordle employs text classification algorithms, such as textCNN and BERT, to analyze and categorize financial text data from forums.
- **Visualization**: BiverWordle combines various visualization techniques, including K-Line diagrams, Theme River, and Word Cloud, to represent stock trends and sentiments.

## DataSet and Checkpoints

- **Data Source**: The data source is from [互动易](http://irm.cninfo.com.cn/), provided by Prof. LI Yuelei, Tianjin University.
  We identified eight categories within the data:

  | **Index** | **Category** | **Index** | **Category** |
  | :---: | :---: | :---: | :---: |
  | 0 | 宏观政策 (Macroeconomic Policy) | 4 | 产品及服务 (Products and Services) |
  | 1 | 业务动态 (Business Dynamics) | 5 | 股票情况及政策 (Stock Status and Policy) |
  | 2 | 收益能力 (Earning Ability) | 6 | 股票价格波动 (Stock Price and Volatility) |
  | 3 | 财务情况 (Financial Situation) | 7 | 其他 (Others) |

- **Checkpoints**: The checkpoints are available in [Google Drive](https://drive.google.com/drive/folders/1RFiNWpuEYn4JRTJFYmAz6n27v46yPPbz?usp=drive_link)

## Text Processing Results
The dataset is partitioned into 11,915 texts for training, 3,910 for testing, and 3,910 for validation, adhering to a 6:2:2 ratio.
We use three models: **textCNN**, **BERT**, and **Voting**. Here are the experiment results for these models:

| Model | Category | Precision | Recall | F1-score |
| :---: | :---: | :---: | :---: | :---: |
| textCNN | 宏观政策 (Macroeconomic Policy) | 0.614 | 0.433 | 0.508 |
| textCNN | 业务动态 (Business Dynamics) | 0.598 | 0.671 | 0.632 |
| textCNN | 收益能力 (Earning Ability) | 0.542 | 0.522 | 0.532 |
| textCNN | 财务情况 (Financial Situation) | 0.748 | 0.593 | 0.661 |
| textCNN | 产品及服务 (Products and Services) | 0.651 | 0.736 | 0.691 |
| textCNN | 股票情况及政策 (Stock Status and Policy) | 0.676 | 0.686 | 0.681 |
| textCNN | 股票价格波动  (Stock Price and Volatility) | 0.611 | 0.627 | 0.619 |
| textCNN | 其他 (Others) | 0.556 | 0.504 | 0.529 |
| BERT | 宏观政策 (Macroeconomic Policy) | 0.526 | 0.514 | 0.520 |
| BERT | 业务动态 (Business Dynamics) | 0.609 | 0.650 | 0.629 |
| BERT | 收益能力 (Earning Ability) | 0.459 | 0.659 | 0.541 |
| BERT | 财务情况 (Financial Situation) | 0.669 | 0.605 | 0.636 |
| BERT | 产品及服务 (Products and Services) | 0.620 | 0.712 | 0.662 |
| BERT | 股票情况及政策 (Stock Status and Policy) | 0.761 | 0.626 | 0.687 |
| BERT | 股票价格波动  (Stock Price and Volatility) | 0.578 | 0.650 | 0.612 |
| BERT | 其他 (Others) | 0.611 | 0.410 | 0.491 |
| Voting | 宏观政策 (Macroeconomic Policy) | 0.607 | 0.491 | 0.543 |
| Voting | 业务动态 (Business Dynamics) | 0.620 | 0.697 | 0.656 |
| Voting | 收益能力 (Earning Ability) | 0.530 | 0.622 | 0.572 |
| Voting | 财务情况 (Financial Situation) | 0.717 | 0.622 | 0.666 |
| Voting | 产品及服务 (Products and Services) | 0.649 | 0.758 | 0.699 |
| Voting | 股票情况及政策 (Stock Status and Policy) | 0.742 | 0.660 | 0.699 |
| Voting | 股票价格波动  (Stock Price and Volatility) | 0.617 | 0.648 | 0.643 |
| Voting | 其他 (Others) | 0.628 | 0.479 | 0.543 |

The Accuracy of Each Model:

| Model | Accuracy |
| :---: | :---: |
| textCNN | 0.624 |
| Bert | 0.612 |
| Voting | 0.642 |


## Visualization Tools

Incorporating G2 into BiverWordle allows you to create interactive and visually appealing charts to enhance your analysis and presentation of stock market sentiment. G2 is a powerful visualization library that provides a wide range of chart types and customization options.

To start using G2 in BiverWordle, follow these steps:

1. Clone the BiverWordle repository from GitHub.
2. Install the required dependencies by running the command `npm install` or `yarn install`.
3. Set up the necessary configurations, such as API keys or database connections, as specified in the project documentation.
4. Utilize the G2 library to create your desired charts and visualizations. Refer to the G2 documentation for more details and examples on how to utilize this powerful visualization library.

For more information about G2, visit the [G2 GitHub repository](https://github.com/antvis/g2).

## Getting Started

To start using BiverWordle, follow these steps:

1. Clone the repository: `git clone https://github.com/Brian-Lei-XIA/BiverWordle.git`
2. Navigate to the project directory: `cd BiverWordle`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Navigate to the code directory: `cd ./code/FinancialVis`
5. Start the system using Djongo: `python manage.py runserver`

Make sure you have Python and Djongo installed on your system before running the above commands. You can find more information about installing Djongo in the [Djongo documentation](https://www.djongomapper.com/get-started/).

## Contributing

## Contributing

Contributions to BiverWordle are welcome! If you would like to contribute to the project, please follow these steps:

1. Fork the BiverWordle repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit your code.
4. Push your changes to your forked repository.
5. Submit a pull request to the main BiverWordle repository.

## Future Improvements

We plan to further enhance our system by:

- Employing specialist annotators to redefine tag categories and provide specialized data tagging.
- Exploring additional classification methods and experimenting with new sentiment analysis algorithms.
- Building a financial category-specific corpus for sentiment analysis.
- Integrating new visualization techniques and advanced features to improve the overall effectiveness and intuitiveness of our system.

## Acknowledgements

We would like to thank the contributors and maintainers of the libraries and frameworks used in this project, as well as the financial text forums and datasets that provided valuable data for analysis.

## Contact

For any inquiries or questions, please contact the BiverWordle team at [brianleixia@connect.hku.hk](mailto:brianleixia@connect.hku.hk).

Enjoy using BiverWordle and happy visualizing!

## Citation
If you find our paper&tool interesting and useful, please feel free to give us a star and cite us through: 
```latex
@inproceedings{10.1145/3615522.3615541,
author = {Xia, Lei and Gao, Yi-Ping and Lin, Le and Chen, Yu-Xi and Zhang, Kang},
title = {BiverWordle: Visualizing Stock Market Sentiment with Financial Text Data and Trends},
booktitle = {Proceedings of the 16th International Symposium on Visual Information Communication and Interaction},
year = {2023},
url = {https://doi.org/10.1145/3615522.3615541},
doi = {10.1145/3615522.3615541},
}
```
## License

BiverWordle is released under the [MIT License](LICENSE).
