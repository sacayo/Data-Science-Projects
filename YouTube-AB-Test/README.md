# YouTube Video Engagement A/B-Test

## Collaborators
[Tracy Volz]()
[Max Dietrich]()
[Tony Gibbons]()
[Ted Johnson]()

## Objective
The project aimed to determine whether YouTube videos experience increased engagement when comments discuss hot-button topics such as Bitcoin. We hypothesized that adding comments containing the word "Bitcoin" to YouTube videos would increase viewer engagement over a short period, potentially due to bot engagement, YouTube's algorithm, or a "shock" factor.

## Methodology
We designed and executed a randomized controlled trial with a multi-level design:

Data Collection: Utilized YouTube Data API to compile a list of videos.
Experimental Design: Implemented simple random assignment to treatment (7 conditions) or control groups.
Treatment: Injected neutral Bitcoin-related comments into videos in the treatment group.
Timeline: Conducted over one month, with a two-week treatment intervention.
Tools: Python for data collection and R for statistical modeling and analysis.

The experiment involved three levels of randomization:

Group assignment (treatment/control)
Experimenter assignment
Comment assignment (for treatment group)

We collected two datasets:

Video engagement data (357 rows)
Comment data (~157,000 rows)

## Results
Initial analysis suggested significant effects on the difference in differences (DiD) of views and comments. However, after applying rigorous statistical methods including:

Bonferroni correction for multiple comparisons
Outlier analysis
Consideration of data distributions (heavy right skew)

We found that the statistical significance of the effects diminished. Key findings include:

No statistically significant effect detected on engagement metrics after corrections.
Outliers had a substantial impact on results, highlighting the importance of thorough data distribution analysis in A/B testing.

#### Significance/Impact
This project demonstrates the complexity of conducting and analyzing A/B tests in real-world scenarios. It highlights the importance of:

Rigorous experimental design in causal inference studies.
Thorough statistical analysis, including corrections for multiple comparisons and outlier handling.
Consideration of data distributions in interpreting results.
The challenges of working with social media data and engagement metrics.

The project provides valuable insights into the nuances of A/B testing in social media contexts and the potential pitfalls in interpreting seemingly significant results. These lessons are directly applicable to data-driven decision making in digital marketing, content strategy, and platform algorithm development.
