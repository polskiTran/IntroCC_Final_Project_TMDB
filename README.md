# Intro Cloud Computing Final Project - TMDB Analysis

## Abstract
Perform movie analysis on TMDB dataset and present the analytics and machine learning prediction through streamlit

## Streamlit pages
1. overview - project abstract + data metadata
2. sample data pull inspect
3. data analytics 
4. ML prediction

### Analytics page
- Average ROI for each genres (bar chart)
- Top director by avg vote avg and avg movie budget (line chart on bar chart)
- ROI scatter plot to draw 3 regions: hit, average, flop categories

### ML prediction page

Input features:
- genres (one hot encoding)
- budget (million dollars)
- release month (1-12)
- director (target encoding)
- production (target encoding)
- 1 lead cast (target encoding)

Targets (2 models):
- Model A: Revenue (million dollars)
- Model B: User rating
