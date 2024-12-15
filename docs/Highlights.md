### Highlights
 Unlike many open-source packages for outlier detection, PyCatcher provides several distinctive features:
* **Automatic Model Selection:** 
PyCatcher automatically detects whether to use an additive or multiplicative
decomposition model, ensuring the most accurate detection of outliers based on the characteristics of your data.
* **Dynamic Method Selection Based on Data Size:**
PyCatcher seamlessly switches between Seasonal Trend Decomposition (for datasets spanning at least two years) and
Inter Quartile Range (IQR) for shorter time periods, offering flexibility without manual intervention.
* **Wide Time Frequency Support:**
Supports multiple time-series frequencies — including daily, weekly, monthly, and quarterly data—without requiring 
users to pre-process or adjust their datasets.
* **Choice of Different Seasonal Trend Algorithms:** Support for outlier detection using various Seasonal Trend 
Decomposition algorithms (Classic; STL; MSTL).
* **Integrated Diagnostics:** PyCatcher includes comprehensive diagnostic tools, enabling users to visualize outliers, 
trends 
and seasonal patterns, evaluate data stationarity, and analyze decomposition results.
* **User Interface:** Availability of a simple user interface for the users to upload file for outlier detection using IQR.
