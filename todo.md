# To improve
- prompt
- generator and viewr for common bugs
- How pipeline works (how output is saved, is it saved in a database, json? will it be usefull for creating the final database?)
- Are results saved directly, one by one or in a batch? (so if error happens, progress is not lost)
- should we include presistant errors in database as bad examples or discard them entirly

# To implement
- Auto logging for each run to compare runs for debugging and comparison (prompt, input, output, settings, model, tempratrue, etc)
- manual/auto evaluation of runs
- A error handeller and follow up prompt to correct the error (e.g. if there is an error, the error and code to be send to model to correct code, with a limit of n retries )
- Fixing error by either (send only python code and error) or (send python code + error + entire history of conversation to better understand the context, but this will be expensive)
- should we make one complete run for each data sample (result -> check whether there is an error -> fix error -> run again then save), or run all data samples, then check those with errors and fix them:
  - Per-sample loop: run → validate → fix → rerun → save
  - Or: run all → validate → fix failures only
