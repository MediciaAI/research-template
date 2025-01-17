# ${GITHUB_REPOSITORY_NAME}

Welcome to MediciaSAFELY's template for setting up a new research study. 

This is the code and configuration for ${GITHUB_REPOSITORY_NAME}.

The instructions in this document help to configure the project according to the your needs. 

The main configuration file is called [project.yaml](https://github.com/MediciaAI/research-template/blob/main/project.yaml). It defines the project pipeline in a series of actions. An action is a logical unit. 

Assuming that the target of the project is to do 3 things:

1. Extract data from MediciaSAFELY database.
2. Use the data to train a machine learning model.
3. Test the trained model. 

This project should have 3 actions. This is an example of the `project.yaml` file. The 3 actions are:

1. generate_study_population
2. train_model
3. predict_model

The user is free to chose the names for these actions.

```yaml
version: '3.0'

actions:
  generate_study_population:
    run: cohortextractor:latest generate_cohort --generate-cohort-by-sql <SQL-file> --database-url=<DB-URL>
    outputs:
      highly_sensitive:
        cohort: output/input.csv
  train_model:
    run: python:latest python analysis/train_model.py
    needs: [generate_study_population]
    outputs:
      moderately_sensitive:
        cohort: output/regression_model.joblib
  predict_model:
    run: python:latest python analysis/model_predict.py
    needs: [train_model]
    outputs:
      moderately_sensitive:
        cohort: output/stats.txt
```

Each action has a `run` command which specifies how the action is executed to generate the output file. A Docker image is used by each action. So, make sure Docker is installed properly.

The image name is specified at the beginning of the `run` command. For example, the image is `cohortextractor` for the first action. Note that the pipeline must start with `cohortextractor` as the first action. The other 2 actions use `python` as the Docker image. The `:latest` after each Docker image name just specifies which version of the image is used.

After the image name and its version, there comes the command that should be executed. For the first action, the command is as follows:

```
generate_cohort --generate-cohort-by-sql <SQL-file> --database-url=<DB-URL>
```

It just asks for generating a cohort based on the SQL queries in the `<SQL-file>` while the database used is accessible through the URL `<DB-URL>`.

Each action generates an output file. The file generated by the `generate_study_population` action is `output/input.csv`.

For the second action, the command is:

```
python analysis/train_model.py
```

It simply asks to execute a Python script named `train_model.py` that exists in the `analysis` folder. So, make sure the Python script is available there. The output file is `output/regression_model.joblib` which is the saved model.

Because the `train_model` action uses the data generated by the `generate_study_population` action, it is listed as a dependency. 

```
needs: [generate_study_population]
```

The third action is similar to the second action.

## Submit the Project for Execution at MediciaSAFELY

To run this project is the secure MediciaSAFELY environment, then use your GitHub account to login to MediciSAFELY at [this link](http://rcdemo.medicia.ai:8000/login): http://rcdemo.medicia.ai:8000/login. Just click the `Sign in with GitHub` button.

Once you logged into MediciaSAFELY, click on your GitHub account icon at the top-right corner to trigger a menu and then click `Applications`: http://rcdemo.medicia.ai:8000/applications. Then click the `Start a new project` button. Fill the project form and click submit at the end.

Once the project is submitted, then you can import your GitHub research study project and run its actions against MediciaSAFELY.

## Run at Gitpod

You can run this project via [Gitpod](https://gitpod.io) in a web browser by clicking on this badge: [![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/${GITHUB_REPOSITORY})

* Raw model outputs, including charts, crosstabs, etc, are in `released_outputs/`
* If you are interested in how we defined our code lists, look in the [codelists folder](./codelists/).
* Developers and epidemiologists interested in the framework should review [the OpenSAFELY documentation](https://docs.opensafely.org)

## Licences

As standard, research projects have a MIT license. 
