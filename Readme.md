# Multi-Model Chat using Snowpark Async

Submits prompts to multiple LLMs using Streamlit Chat Widgets, Snowflake's Complete function through Snowpark Async Methods.



## Deployment Option 1:

Create Streamlit app using Snowsight and replace default app code with code from `streamlit_app.py` file. Include packages using package picker in SiS UI. 

## Deployment Option 2:

1- Create a Snowflake Stage with Directory enabled. 
```sql
CREATE OR REPLACE STAGE DB.ST_APPS.STAGE_NAME
DIRECTORY=(ENABLE=TRUE);
```
2- Upload the `streamli_app.py` and `environment.yml` files into the root of the stage. 

3- Create Streamlit app using the code below. 
```sql
CREATE STREAMLIT ASYNC_MULTI_MODEL_CHAT
  ROOT_LOCATION = '@DB.ST_APPS.STAGE_NAME'
  MAIN_FILE = 'streamlit_app.py'
  QUERY_WAREHOUSE = STREAMLIT_XS;
```

## Basic Process:

```mermaid
graph TB;

step1[Select Models]
step2[Submit Prompt]
step3[Compose Prompt]
step4[For N models in Models] 
step5[Submit Final Prompt]
step6[Add to Prompt History]
step7{Submit all?}
step9[Render N number of columns]
step10[For N models in Models] 
step11{is async_job}
step12{is_done}
step13[mark as pending]
step14[replace async job with final response]
step15[display results]
step16[display waiting placeholder]
step17{any pending?}
step18[wait 5 seconds]
step19[[done]]

step1-->step2-->step3-->step4-->step5-->step6-->step7

step7--N-->step4
step7--Y-->step9
step9-->step10
step10-->step11
step11--N-->step15
step11--Y-->step12
step12--Y-->step14-->step15
step12--N-->step13-->step16
step16-->step17
step15-->step17
step17--N-->step19
step17--Y-->step18-->step10
```