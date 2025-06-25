# 🚲 ValenBisi Route App

## Description

An intelligent web application for Valencia's public bicycle sharing system (ValenBisi). Combines bike availability forecasting, route optimization, and an RAG-based chatbot for querying system policies, all within a user-friendly interface.

## Motivation

ValenBisi’s official platform lacks predictive features and quick access to usage rules. This tool leverages historical data, weather inputs, and AI models to enhance trip planning and policy lookup.

## Key Features

- 🔮 **Availability Forecasting:** LightGBM model trained on station usage and meteorological data to predict bike and dock availability.
- 🗺️ **Route Optimization:** A\* algorithm that balances travel distance, estimated times, and station reliability for optimal routes.
- 🤖 **RAG Chatbot:** Retrieval-Augmented Generation system that ingests the official terms & conditions PDF and delivers precise, context-aware answers.
- 🖥️ **Web Interface:** Built with Streamlit, it offers:
  - Date and time selector for availability maps.
  - Route planner with origin, destination, and maximum travel time inputs.
  - Integrated chatbot for policy inquiries.

## User targets

- **ValenBisi Riders** seeking advanced trip planning tools.
- **Visitors and Tourists** needing quick policy clarifications and route suggestions.
- **Developers and Urban Planners** interested in smart mobility solutions.

## Project Structure

```text
├── app.py
├── data
│   ├── bike_stations.csv
│   ├── dataset.csv
│   ├── distance_matrix.csv
│   ├── documents
│   │   ├── CGAUS_en_valenbisi.pdf
│   │   └── FAQ-valenbici.csv
│   ├── duration_matrix.csv
│   ├── duration_w_matrix.csv
│   └── may_with_predictions.parquet
├── figures
├── images_4_app
│   ├── logo-contract.png
│   └── valenbici_bici.jpeg
├── model
│   ├── best_params.json
│   ├── final_model.pkl
│   └── retrained_model.pkl
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── best_lgbm_optuna.py
│   ├── ChatBot.py
│   ├── config.py
│   ├── extract_data_from_ceferra_repo.py
│   ├── ors_cycling.py
│   ├── prepare_data4ml.py
│   ├── rag_chat_aux.py
│   ├── routes.ipynb
│   ├── routes.py
│   ├── style.css
│   ├── train_test_lgbm.py
│   ├── utils.py
│   ├── vectorstore_builder.py
│   ├── visuals.ipynb
│   └── weather_scraper.py
├── uv.lock
└── vectordb
    ├── index.faiss
    └── index.pkl
```

## Local Setup

1. **Clone the repository**
   ```bash
   git clone git@github.com:andreucs/smart-city.git
```
2.  **Create and activate a virtual environment**
      ```bash
      uv sync
      source .venv/bin/activate
      ```
3. Install the dependencies
```bash
   uv sync
```

## Future Enhancements

- 📈 Real-time model retraining to avoid concept drift.
- 🌲 Climate-aware routing using weather forecasts and tree cover data.
- 🎮 Gamification features to incentivize balanced station usage.
- ⚛️ Migrating the frontend to React or Vue for a richer user experience.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## Website
You can access the app at [https://smart-city-bj5jkxaufvrywmxsbwvz4p.streamlit.app/](https://smart-city-bj5jkxaufvrywmxsbwvz4p.streamlit.app/)

*Developed by A-squared team*

Andreu Bonet Pavia
Anna Gil Moliner

