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
      <td>22.86</td>
      <td>0.51</td>
      <td>0.32</td>
      <td>0.50</td>
      <td>49.51</td>
      <td>48.01</td>
      <td>0.52</td>
      <td>0.83</td>
      <td>1.35</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>16.88</td>
      <td>0.45</td>
      <td>0.26</td>
      <td>0.44</td>
      <td>35.37</td>
      <td>34.48</td>
      <td>0.40</td>
      <td>0.81</td>
      <td>0.77</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>22.09</td>
      <td>0.49</td>
      <td>0.30</td>
      <td>0.48</td>
      <td>41.13</td>
      <td>40.11</td>
      <td>0.45</td>
      <td>0.83</td>
      <td>0.89</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>25.20</td>
      <td>0.52</td>
      <td>0.32</td>
      <td>0.51</td>
      <td>45.16</td>
      <td>43.99</td>
      <td>0.49</td>
      <td>0.84</td>
      <td>1.00</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>26.84</td>
      <td>0.53</td>
      <td>0.34</td>
      <td>0.52</td>
      <td>47.86</td>
      <td>46.63</td>
      <td>0.52</td>
      <td>0.84</td>
      <td>1.10</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>26.63</td>
      <td>0.54</td>
      <td>0.35</td>
      <td>0.53</td>
      <td>49.67</td>
      <td>48.36</td>
      <td>0.53</td>
      <td>0.85</td>
      <td>1.16</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>25.55</td>
      <td>0.54</td>
      <td>0.35</td>
      <td>0.53</td>
      <td>50.05</td>
      <td>48.74</td>
      <td>0.54</td>
      <td>0.84</td>
      <td>1.21</td>
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
      <td>16.93</td>
      <td>0.38</td>
      <td>0.24</td>
      <td>0.37</td>
      <td>36.67</td>
      <td>35.56</td>
      <td>0.39</td>
      <td>0.61</td>
      <td>1.35</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>21.92</td>
      <td>0.58</td>
      <td>0.34</td>
      <td>0.57</td>
      <td>45.94</td>
      <td>44.78</td>
      <td>0.52</td>
      <td>1.05</td>
      <td>0.77</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>24.82</td>
      <td>0.55</td>
      <td>0.34</td>
      <td>0.54</td>
      <td>46.21</td>
      <td>45.07</td>
      <td>0.51</td>
      <td>0.93</td>
      <td>0.89</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>25.20</td>
      <td>0.52</td>
      <td>0.32</td>
      <td>0.51</td>
      <td>45.16</td>
      <td>43.99</td>
      <td>0.49</td>
      <td>0.84</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>24.40</td>
      <td>0.48</td>
      <td>0.31</td>
      <td>0.47</td>
      <td>43.51</td>
      <td>42.39</td>
      <td>0.47</td>
      <td>0.76</td>
      <td>1.10</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>22.96</td>
      <td>0.47</td>
      <td>0.30</td>
      <td>0.46</td>
      <td>42.82</td>
      <td>41.69</td>
      <td>0.46</td>
      <td>0.73</td>
      <td>1.16</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>21.12</td>
      <td>0.45</td>
      <td>0.29</td>
      <td>0.44</td>
      <td>41.36</td>
      <td>40.28</td>
      <td>0.45</td>
      <td>0.69</td>
      <td>1.21</td>
      <td>0.65</td>
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
      <td>38.81</td>
      <td>0.67</td>
      <td>0.49</td>
      <td>0.64</td>
      <td>66.20</td>
      <td>63.65</td>
      <td>0.64</td>
      <td>0.88</td>
      <td>1.08</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>28.95</td>
      <td>0.58</td>
      <td>0.42</td>
      <td>0.55</td>
      <td>52.94</td>
      <td>50.80</td>
      <td>0.48</td>
      <td>0.85</td>
      <td>0.75</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>32.62</td>
      <td>0.62</td>
      <td>0.44</td>
      <td>0.58</td>
      <td>56.93</td>
      <td>54.69</td>
      <td>0.54</td>
      <td>0.86</td>
      <td>0.84</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>36.46</td>
      <td>0.65</td>
      <td>0.47</td>
      <td>0.62</td>
      <td>61.34</td>
      <td>58.95</td>
      <td>0.59</td>
      <td>0.88</td>
      <td>0.93</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>38.19</td>
      <td>0.66</td>
      <td>0.48</td>
      <td>0.63</td>
      <td>63.19</td>
      <td>60.78</td>
      <td>0.61</td>
      <td>0.88</td>
      <td>0.98</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>39.09</td>
      <td>0.67</td>
      <td>0.49</td>
      <td>0.63</td>
      <td>64.53</td>
      <td>62.08</td>
      <td>0.62</td>
      <td>0.88</td>
      <td>1.01</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>38.63</td>
      <td>0.67</td>
      <td>0.49</td>
      <td>0.63</td>
      <td>64.82</td>
      <td>62.32</td>
      <td>0.63</td>
      <td>0.88</td>
      <td>1.03</td>
      <td>0.93</td>
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
      <td>35.94</td>
      <td>0.62</td>
      <td>0.45</td>
      <td>0.59</td>
      <td>61.30</td>
      <td>58.94</td>
      <td>0.59</td>
      <td>0.81</td>
      <td>1.08</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>38.60</td>
      <td>0.77</td>
      <td>0.56</td>
      <td>0.73</td>
      <td>70.59</td>
      <td>67.73</td>
      <td>0.64</td>
      <td>1.13</td>
      <td>0.75</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>38.83</td>
      <td>0.74</td>
      <td>0.52</td>
      <td>0.69</td>
      <td>67.77</td>
      <td>65.11</td>
      <td>0.64</td>
      <td>1.02</td>
      <td>0.84</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>39.20</td>
      <td>0.70</td>
      <td>0.51</td>
      <td>0.67</td>
      <td>65.96</td>
      <td>63.39</td>
      <td>0.63</td>
      <td>0.95</td>
      <td>0.93</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>38.97</td>
      <td>0.67</td>
      <td>0.49</td>
      <td>0.64</td>
      <td>64.48</td>
      <td>62.02</td>
      <td>0.62</td>
      <td>0.90</td>
      <td>0.98</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>38.70</td>
      <td>0.66</td>
      <td>0.49</td>
      <td>0.62</td>
      <td>63.89</td>
      <td>61.47</td>
      <td>0.61</td>
      <td>0.87</td>
      <td>1.01</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>37.50</td>
      <td>0.65</td>
      <td>0.48</td>
      <td>0.61</td>
      <td>62.93</td>
      <td>60.50</td>
      <td>0.61</td>
      <td>0.85</td>
      <td>1.03</td>
      <td>0.56</td>
    </tr>
  </tbody>
</table>
</div>