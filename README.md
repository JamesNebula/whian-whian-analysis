## Data Quality Note

Raw processing detected a maximum canopy height of 137.3m — unrealistic for Australian 
rainforest (typical max: 60m). In production workflows, this would trigger:

1. Outlier filtering (>60m threshold)
2. DTM gap validation in high-height areas
3. Point cloud noise assessment

This quality check demonstrates the importance of scientific skepticism in spatial analysis.
