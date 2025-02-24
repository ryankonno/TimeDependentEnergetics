import os 
from datetime import datetime 

def createSavePath(params):
    today = datetime.now()
    run_id = today.strftime('%Y%m%d_%H%m%s') 

    params['run_id'] = run_id
    params['MU_model']['run_id'] = run_id

    results_save_dir = './Results/' + run_id
    os.mkdir(results_save_dir)

    figures_save_dir = './Figures/' + run_id
    os.mkdir(figures_save_dir)

    return params