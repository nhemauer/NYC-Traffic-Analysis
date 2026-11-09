# NYC-Traffic-Analysis

Ask questions in plain English about NYC yellow taxi trips from 2011 to today. A Gemini agent writes and runs BigQuery SQL to answer, estimates fares with a random forest, and forecasts future trips with histogram gradient boosting.

**Here:** [www.nickhemauer.com/taxi](https://www.nickhemauer.com/taxi)

## ETL

- **Old data:** 2011–2022 trips come from the public NYC taxi dataset in BigQuery.
- **Recent data:** a Cloud Run job, started weekly by Cloud Scheduler, downloads and loads the monthly trip files published by the NYC Taxi & Limousine Commission.
- **Two tables:**
  - A 1-in-1,000 sample of individual trips, to keep the data reasonable for local use.
  - Hourly totals of trips and fares for every day to complement the overall sample and provide accurate cumulative answers.

## Agent

- Built on Gemini 3.5 Flash and querying BigQuery from natural-language questions.
- Has the fare and forecast models below as tools.
- **Faster responses:** the table schemas and the current date are written into the agent's instructions, so it doesn't have to look up the tables for every request.
- **CI/CD:** a Cloud Build trigger redeploys the agent whenever the code changes in the repo.

## Trained and Deployed Models

### Fare prediction (random forest)

- Predicts the metered fare from the year, trip distance, duration, passenger count, pickup hour, day of the week, and month.
- Tuned with a small grid search and 3-fold cross-validation.
- Every prediction is saved to a BigQuery table to record the questions asked and the model's answers.

### Trip forecasts (histogram gradient boosting)

- Forecasts trip counts, total fares, and the busiest times of day.
- Trained on data from January 2023 onward, when NYC last changed the fare rates, which also leaves out COVID-era travel.
- Forecasts can reach at most 2 years past the latest data.

## Website

- Hosted on Firebase, and GitHub Actions redeploys it whenever the [site's repo](https://github.com/nhemauer/Personal-Website) changes.
