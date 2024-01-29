# Translate and Compress
Download data from OpenSub/EuroParl from the `/data` folder. Each folder has separate scripts for this.
- We include a batch downloader for several languages. For specific languages:
    - `python download_langs.py en fr` for english--french parallel data


Fine-tuning of models can be seen in `finetune.py`. 

Printouts of experiments on both in-domain (opensubtitles) and out-of-domain (europarl) are in their respective jupyter notebooks in the root dir. See a temporary sample below. Note that the data is randomly sampled from the original distribution, which favors higher compression ratios. However, it is applicable as real-world data.

## OpenSubtitles - French
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bleu</th>
      <th>r1</th>
      <th>r2</th>
      <th>rl</th>
      <th>chrF</th>
      <th>chrf++</th>
      <th>meteor</th>
      <th>bert_f1</th>
      <th>len_ratio</th>
      <th>normalized_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>2.07</td>
      <td>0.06</td>
      <td>0.01</td>
      <td>0.06</td>
      <td>15.12</td>
      <td>13.91</td>
      <td>0.15</td>
      <td>0.75</td>
      <td>1.22</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>17.55</td>
      <td>0.46</td>
      <td>0.25</td>
      <td>0.45</td>
      <td>35.20</td>
      <td>34.55</td>
      <td>0.40</td>
      <td>0.81</td>
      <td>0.75</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>24.80</td>
      <td>0.51</td>
      <td>0.30</td>
      <td>0.50</td>
      <td>42.95</td>
      <td>42.10</td>
      <td>0.47</td>
      <td>0.84</td>
      <td>0.88</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>27.49</td>
      <td>0.54</td>
      <td>0.33</td>
      <td>0.53</td>
      <td>46.54</td>
      <td>45.61</td>
      <td>0.52</td>
      <td>0.85</td>
      <td>0.99</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>29.03</td>
      <td>0.56</td>
      <td>0.35</td>
      <td>0.55</td>
      <td>49.85</td>
      <td>48.80</td>
      <td>0.54</td>
      <td>0.85</td>
      <td>1.07</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>28.47</td>
      <td>0.55</td>
      <td>0.35</td>
      <td>0.54</td>
      <td>50.37</td>
      <td>49.27</td>
      <td>0.54</td>
      <td>0.85</td>
      <td>1.13</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>27.27</td>
      <td>0.55</td>
      <td>0.35</td>
      <td>0.54</td>
      <td>51.00</td>
      <td>49.82</td>
      <td>0.55</td>
      <td>0.85</td>
      <td>1.18</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>

## Normalized on compression length:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bleu</th>
      <th>r1</th>
      <th>r2</th>
      <th>rl</th>
      <th>chrF</th>
      <th>chrf++</th>
      <th>meteor</th>
      <th>bert_f1</th>
      <th>len_ratio</th>
      <th>normalized_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>1.70</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>12.39</td>
      <td>11.40</td>
      <td>0.12</td>
      <td>0.61</td>
      <td>1.22</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>23.40</td>
      <td>0.61</td>
      <td>0.33</td>
      <td>0.60</td>
      <td>46.93</td>
      <td>46.07</td>
      <td>0.53</td>
      <td>1.08</td>
      <td>0.75</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>28.18</td>
      <td>0.58</td>
      <td>0.34</td>
      <td>0.57</td>
      <td>48.81</td>
      <td>47.84</td>
      <td>0.53</td>
      <td>0.95</td>
      <td>0.88</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>27.77</td>
      <td>0.55</td>
      <td>0.33</td>
      <td>0.54</td>
      <td>47.01</td>
      <td>46.07</td>
      <td>0.53</td>
      <td>0.86</td>
      <td>0.99</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>27.13</td>
      <td>0.52</td>
      <td>0.33</td>
      <td>0.51</td>
      <td>46.59</td>
      <td>45.61</td>
      <td>0.50</td>
      <td>0.79</td>
      <td>1.07</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>25.19</td>
      <td>0.49</td>
      <td>0.31</td>
      <td>0.48</td>
      <td>44.58</td>
      <td>43.60</td>
      <td>0.48</td>
      <td>0.75</td>
      <td>1.13</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>23.11</td>
      <td>0.47</td>
      <td>0.30</td>
      <td>0.46</td>
      <td>43.22</td>
      <td>42.22</td>
      <td>0.47</td>
      <td>0.72</td>
      <td>1.18</td>
      <td>0.85</td>
    </tr>
  </tbody>
</table>
</div>


## EuroParl - French
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bleu</th>
      <th>r1</th>
      <th>r2</th>
      <th>rl</th>
      <th>chrF</th>
      <th>chrf++</th>
      <th>meteor</th>
      <th>bert_f1</th>
      <th>len_ratio</th>
      <th>normalized_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>1.62</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>21.43</td>
      <td>17.61</td>
      <td>0.07</td>
      <td>0.76</td>
      <td>1.07</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>30.82</td>
      <td>0.60</td>
      <td>0.42</td>
      <td>0.56</td>
      <td>54.56</td>
      <td>52.47</td>
      <td>0.50</td>
      <td>0.85</td>
      <td>0.75</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>33.64</td>
      <td>0.62</td>
      <td>0.44</td>
      <td>0.59</td>
      <td>57.96</td>
      <td>55.77</td>
      <td>0.54</td>
      <td>0.86</td>
      <td>0.83</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>37.21</td>
      <td>0.66</td>
      <td>0.47</td>
      <td>0.62</td>
      <td>61.81</td>
      <td>59.54</td>
      <td>0.59</td>
      <td>0.88</td>
      <td>0.91</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>38.73</td>
      <td>0.67</td>
      <td>0.48</td>
      <td>0.63</td>
      <td>63.67</td>
      <td>61.34</td>
      <td>0.62</td>
      <td>0.88</td>
      <td>0.97</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>39.69</td>
      <td>0.67</td>
      <td>0.49</td>
      <td>0.64</td>
      <td>65.09</td>
      <td>62.71</td>
      <td>0.63</td>
      <td>0.89</td>
      <td>1.01</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>38.59</td>
      <td>0.67</td>
      <td>0.49</td>
      <td>0.64</td>
      <td>65.25</td>
      <td>62.79</td>
      <td>0.64</td>
      <td>0.89</td>
      <td>1.03</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>

### Normalized on compression length:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bleu</th>
      <th>r1</th>
      <th>r2</th>
      <th>rl</th>
      <th>chrF</th>
      <th>chrf++</th>
      <th>meteor</th>
      <th>bert_f1</th>
      <th>len_ratio</th>
      <th>normalized_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>1.51</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>20.03</td>
      <td>16.46</td>
      <td>0.07</td>
      <td>0.71</td>
      <td>1.07</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>41.09</td>
      <td>0.80</td>
      <td>0.56</td>
      <td>0.75</td>
      <td>72.75</td>
      <td>69.96</td>
      <td>0.67</td>
      <td>1.13</td>
      <td>0.75</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>40.53</td>
      <td>0.75</td>
      <td>0.53</td>
      <td>0.71</td>
      <td>69.83</td>
      <td>67.19</td>
      <td>0.65</td>
      <td>1.04</td>
      <td>0.83</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>40.89</td>
      <td>0.73</td>
      <td>0.52</td>
      <td>0.68</td>
      <td>67.92</td>
      <td>65.43</td>
      <td>0.65</td>
      <td>0.97</td>
      <td>0.91</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>39.93</td>
      <td>0.69</td>
      <td>0.49</td>
      <td>0.65</td>
      <td>65.64</td>
      <td>63.24</td>
      <td>0.64</td>
      <td>0.91</td>
      <td>0.97</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>39.30</td>
      <td>0.66</td>
      <td>0.49</td>
      <td>0.63</td>
      <td>64.45</td>
      <td>62.09</td>
      <td>0.62</td>
      <td>0.88</td>
      <td>1.01</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>37.47</td>
      <td>0.65</td>
      <td>0.48</td>
      <td>0.62</td>
      <td>63.35</td>
      <td>60.96</td>
      <td>0.62</td>
      <td>0.86</td>
      <td>1.03</td>
      <td>0.87</td>
    </tr>
  </tbody>
</table>
</div>