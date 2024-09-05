1. Comments should be provided sufficiently
2. Grade -> documentation, experiment, implementation
3. Deadline: 2 weeks
4. Submit: github link contains: jupyter notebook, README.md, and folder app (web application)

Task 1: Prepare dataset:
TODO:
1. Perform loading
2. EDA
3. preprocessing
4. model selection
5. inference (use best practices)

Coding considerations:
1. Feature **owner** - map First owner to 1, ..., Test drive car to 5 #DP
2. Feature fuel - remove all rows with CNG and LPG because CNG and LPG use a different mileage system (km/kg) which is different from kmpl for Diesel and Petrol #DP
3. Feature mileage - remove "kmpl" and convert to float #DP
4. Feature engine - remove "CC" and convert to numerical #DP
5. Feature max power - same as engine #DP
6. Feature brand - take first word and remove other #DP
7. Drop feature torque #DP
8. Test Drive cars are expensive, so delete all samples #DP
9. Feature Selling price - fit log transform -> so at inference return back to original #FE

Task 2: Report:
In the end of the notebook, write 2-3 paragraphs summary (hypothesis).
- Which features are important? Which are not? Why? 
- Which algorithm performs well? Which does not? Why? (here, you haven’t learned about any algorithms yet, but you can search online a bit and start building an intuition)

Task 3: Deployment:
Web-based application that uses model. Deploy model into production. Guide:
The goal of this track is to expose/deploy our model for public use via the web interface. The main scenario:
1. Users enter the domain on browser.
2. (optional) Users may need to navigate a prediction page
3. Instructions of prediction
4. Input form, put in the appropriate data, and click submit
5. Allow null fields for form (fill the missing fields with imputation techniques)
6. Print the result (return API?)
Recommendations: Use Dash by Plotly for one-stop solution.
Folder app: should contain .Dockerfile, docker-compose.yaml, and code folder.

[Dockerized Dash project](https://github.com/chaklam-silpasuwanchai/Machine-Learning/tree/main/Appendix/Appendix%20-%20Dash%20Plotly)


# All mappings
```
owner_map = {

    'owner': {

        "First Owner": 1,

        "Second Owner": 2,

        "Third Owner": 3,

        "Fourth & Above Owner": 4,

        "Test Drive Car": 5,

    }

}

df_copy.replace(owner_map, inplace=True)
```

```
brand_name_map = {'brand': {v:k for k, v in zip(range(1, 33),

        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',

       'Mahindra', 'Tata', 'Chevrolet', 'Fiat', 'Datsun', 'Jeep',

       'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW',

       'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',

       'Kia', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel', 'Peugeot'])

    }

}

df_copy.replace(brand_name_map, inplace=True)
```
