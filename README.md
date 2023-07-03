# BiverWordle: Visualizing Stock Market Sentiment with Financial Text Data and Trends

## Overview

![Overview of visualization method BiverWordle, using example stock 002594 BYD](https://github.com/Brian-Lei-XIA/BiverWordle/blob/main/images/overview.png)

BiverWordle is a novel visualization system for analyzing, visualizing, and verifying the sentiments of stock trends on financial text forums. It combines stock data visualization techniques, such as K-Line diagrams, with text data visualization methods like Theme River and Word Cloud, to provide a comprehensive and intuitive representation of stock market sentiment.

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
  We identified eight categories within the data：
| **Index** | **Category** | **Index** | **Category** |
| :---: | :--- | :---: | :--- |
| 0 | 宏观政策 (Macroeconomic Policy) | 4 | 产品及服务 (Products and Services) |
| 1 | 业务动态 (Business Dynamics) | 5 | 股票情况及政策 (Stock Status and Policy) |
| 2 | 收益能力 (Earning Ability) | 6 | 股票价格波动 (Stock Price and Volatility) |
| 3 | 财务情况 (Financial Situation) | 7 | 其他 (Others) |

- **Checkpoints**: The checkpoints are available in [Google Drive](https://drive.google.com/drive/folders/1RFiNWpuEYn4JRTJFYmAz6n27v46yPPbz?usp=drive_link)

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

## License

BiverWordle is released under the [MIT License](LICENSE).
