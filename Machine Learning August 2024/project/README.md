# Easy Credit Eligibility Checker

This repository contains a web-based prediction application, the project for machine learning course of DSAI program, that integrates Django for the backend framework, Google Generative AI for enhanced interactivity, and a machine learning model for predicting credit eligibility. The application includes form handling, machine learning predictions, and deployment configurations.

## Project Structure

### Key Files

1. **`forms.py`**
   - Defines the `DetectionForm` class, which handles user input for predictions.
   - Includes fields like gender, income, housing, job title, and other relevant features.
   - Utilizes Django forms with custom widgets for styling.

2. **`views.py`**
   - Contains views for rendering templates and handling form submissions.
   - Includes:
     - `IndexView`: Displays the home page.
     - `SuccessView`: Displays the prediction results.
     - `PredictFormView`: Manages form submission, integrates with a machine learning model, and generates predictions.
   - Utilizes a machine learning model dumped to a pickle file (`best_model_pipeline.pkl`) for predictions.
   - Integrates with Google Generative AI (`gemini-1.5-flash-latest`) for dynamic content generation.

3. **`CICD.yml`**
   - Defines the Continuous Integration and Continuous Deployment (CI/CD) pipeline configuration.
   - Likely includes steps for testing, building, and deploying the application.

4. **`docker-compose.yml`**
   - Sets up the environment for containerized application deployment.
   - Defines services, networks, and volumes for the Docker setup.

5. **`Dockerfile`**
   - Provides instructions for building the Docker image for the application.
   - Specifies the base image, dependencies, and application setup steps.
