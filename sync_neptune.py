import neptune.new as neptune
import time

# Create a list of strings with integers between 2 numbers
def list_of_ints(start, end):
    return [str(i) for i in range(start, end)]

# Concatenate s string to all strings in a list
def concat_list(s, l):
    return [s + str(i) for i in l]

run_names = concat_list('AR-', list_of_ints(1, 100))



# Go throug all runs and init, sleep then stop
def run_all(run_names):
    for i in run_names:
        run = neptune.init( project="aksel.s.madsen/artefact-detection",
                            run = i,
                            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYTA4NzcxMy1lYmQ2LTQ3NTctYjRhNC02Mzk1NjdjMWM0NmYifQ==")
        # sleep for 4 seconds
        time.sleep(4)
        run.stop()
        
run = neptune.init(
    project="NTLAB/artifactDetect-ear", 
    run = "AR-32",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYTA4NzcxMy1lYmQ2LTQ3NTctYjRhNC02Mzk1NjdjMWM0NmYifQ==", # your credentials
)
# Upload the local neptune data

run["testing/update_test"].log(1000)
run.sync(wait = True)
run.stop()

        
        
#run_all(run_names)