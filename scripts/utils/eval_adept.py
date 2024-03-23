import pandas as pd
import warnings
import os
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None 

def eval_adept(path):
    net = 'net1'

    # read pickle file
    tf = pd.DataFrame()
    sf = pd.DataFrame()
    af = pd.DataFrame()

    with open(os.path.join(path, 'trialframe.csv'), 'rb') as f:
        tf_temp = pd.read_csv(f, index_col=0)
    tf_temp['net'] = net
    tf = pd.concat([tf,tf_temp])

    with open(os.path.join(path, 'slotframe.csv'), 'rb') as f:
        sf_temp = pd.read_csv(f, index_col=0)
    sf_temp['net'] = net
    sf = pd.concat([sf,sf_temp])

    with open(os.path.join(path, 'accframe.csv'), 'rb') as f:
        af_temp = pd.read_csv(f, index_col=0)
    af_temp['net'] = net
    af = pd.concat([af,af_temp])

    # cast variables
    sf['visible'] = sf['visible'].astype(bool)
    sf['bound'] = sf['bound'].astype(bool)
    sf['occluder'] = sf['occluder'].astype(bool)
    sf['inimage'] = sf['inimage'].astype(bool)
    sf['vanishing'] = sf['vanishing'].astype(bool)
    sf['alpha_pos'] = 1-sf['alpha_pos']
    sf['alpha_ges'] = 1-sf['alpha_ges']

    # scale to percentage
    sf['TE'] = sf['TE'] * 100

    # add surprise as dummy code
    tf['control'] = [('control' in set) for set in tf['set']]
    sf['control'] = [('control' in set)  for set in sf['set']]



    # STATS:
    tracking_error_visible = 0
    tracking_error_occluded = 0
    num_positive_trackings = 0
    mota = 0
    gate_openings_visible = 0
    gate_openings_occluded = 0


    print('Tracking Error ------------------------------')
    grouping = (sf.inimage & sf.bound & ~sf.occluder & sf.control)

    def get_stats(col):
        return f' M: {col.mean():.3} , STD: {col.std():.3}, Count: {col.count()}'

    # When Visible
    temp = sf[grouping & sf.visible]
    print(f'Tracking Error when visible:' + get_stats(temp['TE']))
    tracking_error_visible = temp['TE'].mean()

    # When Occluded
    temp = sf[grouping & ~sf.visible]
    print(f'Tracking Error when occluded:' + get_stats(temp['TE']))
    tracking_error_occluded = temp['TE'].mean()






    print('Positive Trackings ------------------------------')
    # succesfull trackings: In the last visible moment of the target, the slot was less than 10% away from the target
    # determine last visible frame numeric
    grouping_factors = ['net','set','evalmode','scene','slot']
    ff = sf[sf.visible & sf.bound & sf.inimage].groupby(grouping_factors).max()
    ff.rename(columns = {'frame':'last_visible'}, inplace = True)
    sf = sf.merge(ff[['last_visible']], on=grouping_factors, how='left')

    # same for first bound frame
    ff = sf[sf.visible & sf.bound & sf.inimage].groupby(grouping_factors).min()
    ff.rename(columns = {'frame':'first_visible'}, inplace = True)
    sf = sf.merge(ff[['first_visible']], on=grouping_factors, how='left')

    # add dummy variable to sf
    sf['last_visible'] = (sf['last_visible'] == sf['frame'])

    # extract the trials where the target was last visible and threshold the TE
    ff = sf[sf['last_visible']] 
    ff['tracked_pos'] = (ff['TE'] < 10)
    ff['tracked_neg'] = (ff['TE'] >= 10)

    # fill NaN with 0
    sf = sf.merge(ff[grouping_factors + ['tracked_pos', 'tracked_neg']], on=grouping_factors, how='left')
    sf['tracked_pos'].fillna(False, inplace=True)
    sf['tracked_neg'].fillna(False, inplace=True)

    # Aggreagte over all scenes
    temp = sf[(sf['frame']== 1) & ~sf.occluder & sf.control & (sf.first_visible < 20)]
    temp = temp.groupby(['set', 'evalmode']).sum()
    temp = temp[['tracked_pos', 'tracked_neg']]
    temp = temp.reset_index()

    temp['tracked_pos_pro'] = temp['tracked_pos'] / (temp['tracked_pos'] + temp['tracked_neg'])
    temp['tracked_neg_pro'] = temp['tracked_neg'] / (temp['tracked_pos'] + temp['tracked_neg'])
    print(temp)
    num_positive_trackings = temp['tracked_pos_pro']





    print('Mostly Trecked /MOTA ------------------------------')
    temp = af[af.index == 'OVERALL']
    temp['mostly_tracked'] = temp['mostly_tracked'] / temp['num_unique_objects']
    temp['partially_tracked'] = temp['partially_tracked'] / temp['num_unique_objects']
    temp['mostly_lost'] = temp['mostly_lost'] / temp['num_unique_objects']
    print(temp)
    mota = temp['mota']


    print('Openings ------------------------------')
    grouping = (sf.inimage & sf.bound & ~sf.occluder & sf.control)
    temp = sf[grouping & sf.visible]
    print(f'Percept gate openings when visible:' + get_stats(temp['alpha_pos'] + temp['alpha_ges']))
    gate_openings_visible = temp['alpha_pos'].mean() + temp['alpha_ges'].mean()

    temp = sf[grouping & ~sf.visible]
    print(f'Percept gate openings when occluded:' + get_stats(temp['alpha_pos'] + temp['alpha_ges']))
    gate_openings_occluded = temp['alpha_pos'].mean() + temp['alpha_ges'].mean()


    print('------------------------------------------------')
    print('------------------------------------------------')
    str = ''
    str += f'net: {net}\n'
    str += f'Tracking Error when visible: {tracking_error_visible:.3}\n'
    str += f'Tracking Error when occluded: {tracking_error_occluded:.3}\n'
    str += 'Positive Trackings: ' + ', '.join(f'{val:.3}' for val in num_positive_trackings) + '\n'
    str += 'MOTA: ' + ', '.join(f'{val:.3}' for val in mota) + '\n'
    str += f'Percept gate openings when visible: {gate_openings_visible:.3}\n'
    str += f'Percept gate openings when occluded: {gate_openings_occluded:.3}\n'

    print(str)

    # write tstring to file
    with open(os.path.join(path, 'results.txt'), 'w') as f:
        f.write(str)
    

    
if __name__ == "__main__":

    # use argparse to get the path to the results folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()

    # setting path to results folder
    path = args.path
    eval_adept(path)