import uproot
import h5py
import numpy as np
import awkward as ak
import vector
import matplotlib.pyplot as plt
import mplhep as hep
import boost_histogram as bh
from tqdm import tqdm
import scipy

def create_file(wilsons, end = ''):
    file_names = []
    for i in wilsons:
        file_names.append(f'EFT_{i}_higher{end}.root')
        file_names.append(f'EFT_{i}_lower{end}.root')
    return file_names

def construct_array(t, tbar):
    # finds differences of values for each pair of t tbar
    dic = {'rapidity':[], 'phi':[], 'R':[]}
    array = 0
    for i in range(2):
        for j in range(2):
            d_rapidity = t[:,i].rapidity - tbar[:,j].rapidity
            d_R =  t[:,i].deltaR(tbar[:,j])
            d_phi = t[:,i].deltaphi(tbar[:,j])
            dic['rapidity'].append(d_rapidity)
            dic['phi'].append(d_phi)
            dic['R'].append(d_R)
    return dic

def b2(vector1, vector2):
    numerator = vector1.py * vector2.py+ vector1.px * vector2.px
    return numerator/(vector1.p * vector2.p)

def b4(vector1, vector2):
    return vector1.pz*vector2.pz/(vector1.p * vector2.p)

def norm_p(vec):
    vec_p = np.array([vec.x, vec.y, vec.z])
    return vec_p / np.sqrt(np.sum(vec_p**2, axis=0))


def boosting(pair1, pair2_t, pair2_tbar):
    pair2_t = pair2_t.boostCM_of(pair1)
    pair2_t_p = norm_p(pair2_t)
    pair2_tbar = pair2_tbar.boostCM_of(pair1)
    pair2_tbar_p = norm_p(pair2_tbar)
    beam_axis = vector.obj(x=0, y=0, z=1, t=0).boost(pair1)
    beam_axis_p = norm_p(beam_axis)
    a = np.cross(pair2_t_p, pair2_tbar_p, axis=0)
    return a


def form_data(tree):

    t = vector.zip({name:tree[f't_{name}'].array() for name in ['pt', 'eta', 'phi', 'e']})
    tbar = vector.zip({name:tree[f'tbar_{name}'].array() for name in ['pt', 'eta', 'phi', 'e']})


    tt = t[:,0] + t[:,1] #for eta and invariant mass, rapidity
    tbartbar = tbar[:,0] + tbar[:,1] # for eta and invariant mass, rapidity
    tttt = tt + tbartbar
    a = t[:,0]+tbar[:,0]
    b = t[:,0]+tbar[:,1]
    c = t[:,1]+tbar[:,0]
    d = t[:,1]+tbar[:,1]

    ttbars = np.concatenate([a[:,None], b[:,None], c[:,None], d[:,None]], axis=1)
    # each combination of t and tbar

    dic = {}
    differences = construct_array(t, tbar)


    dRs = np.transpose(differences['R'],(1,0))
    arg = np.argmin(dRs, axis=1)
    pair1_t = t[ak.local_index(t) == arg[:,None]//2][:,0]
    pair1_tbar = tbar[ak.local_index(tbar) == arg[:,None]%2][:,0]
    pair2_t = t[ak.local_index(t) == (3-arg[:,None])//2][:,0]
    pair2_tbar = tbar[ak.local_index(tbar) == (3-arg[:,None])%2][:,0]
    ttbar1 = pair1_t + pair1_tbar
    ttbar2 = pair2_t + pair2_tbar

    dic['Ht'] = t[:,0].pt + t[:,1].pt + tbar[:,0].pt + tbar[:,1].pt
    dic['M_tttt'] = tttt.m
    # dic['tt_m'] = tt.m
    # dic['tbartbar_m'] = tbartbar.m
    # dic['greatest_mass_ttbar']  = np.max([ttbars[:,i].m for i in range(4)], axis = 0)
    sort = np.argsort(np.abs(ttbars.eta), axis=1)
    dic['max_eta_ttbar']  = ttbars.eta[sort==0][:,0]
    # dic['tt_eta'] = tt.eta
    # dic['tbartbar_eta'] = tbartbar.eta
    # dic['average_eta_ttbar'] = np.mean([ttbars[:,i].eta for i in range(4)], axis = 0)
    # dic['min_eta_ttbar']  = ttbars.eta[sort==3][:,0]

    
    # dic['tt_drapidity'] = np.abs(t[:,0].rapidity - t[:,1].rapidity)
    # dic['tbartbar_rapidity'] = np.abs(tbar[:,0].rapidity - tbar[:,1].rapidity)
    # dic['greatest_rapidity_ttbar']  = np.max(differences['rapidity'], axis = 0)
    # dic['average_rapidity_ttbar'] = np.mean(differences['rapidity'], axis = 0)
    # dic['min_rapidity'] = np.min(differences['rapidity'], axis = 0)
    
    # dic['tt_phi'] = np.abs(t[:,0].phi - t[:,1].phi)
    # dic['tbartbar_phi'] = np.abs(tbar[:,0].phi - tbar[:,1].phi)
    
    # dic['tt_R'] = t[:,0].deltaR(t[:,1])
    # dic['tbartbar_R'] = tbar[:,0].deltaR(tbar[:,1])
    # dic['greatest_R_ttbar']  = np.max(differences['R'], axis = 0)
    # dic['average_R'] = np.mean(differences['R'], axis = 0)
    # dic['min_R'] = np.min(differences['R'], axis = 0)
    
    # dic['tt_pt'] = tt.pt
    # dic['tbartbar_pt'] = tbartbar.pt
    # dic['greatest_pt']  = np.max([ttbars[:,i].pt for i in range(4)], axis = 0)
    # dic['ttbar_pt_smallest_dR1'] = ttbar1.pt
    # dic['ttbar_pt_smallest_dR2'] = ttbar2.pt
    
    # dic['overall_max_pt'] = np.max([i.pt for i in t + tbar], axis = 0)
    # dic['overall_min_pt'] = np.min([i.pt for i in t + tbar], axis = 0)
    # dic['overall_diff_pt'] = dic['overall_max_pt'] - dic['overall_min_pt']
    
    # dic['tt_alignment'] = b2(t[:,0], t[:,1])
    # dic['tbartbar_alignment'] = b2(tbar[:,0], tbar[:,1])
    # b2s = np.stack((b2(t[:,0],tbar[:,0]), b2(t[:,0],tbar[:,1]), b2(t[:,1],tbar[:,0]), b2(t[:,1],tbar[:,1])), axis = 1)
    b4s = np.stack((b4(t[:,0],tbar[:,0]), b4(t[:,0],tbar[:,1]), b4(t[:,1],tbar[:,0]), b4(t[:,1],tbar[:,1])), axis = 1)
    max_ttbar_pt_mask = np.argsort([i.pt for i in ttbars], axis = 1)==0

    # dic['leading_pt_b2'] = b2s[max_ttbar_pt_mask]
    dic['leading_pt_b4'] = b4s[max_ttbar_pt_mask]
    
    # dic['ttbar1_b2'] = b2(pair1_t, pair1_tbar)
    # dic['ttbar1_b4'] = b4(pair1_t, pair1_tbar)
    # dic['ttbar2_b2'] = b2(pair2_t, pair2_tbar)
    # dic['ttbar2_b4'] = b4(pair2_t, pair2_tbar)
    

    
    # a = boosting(ttbar1, pair2_t, pair2_tbar)
    # dic['boosted_cross_ttbar1_x'] = a[0]
    # dic['boosted_cross_ttbar1_y'] = a[1]
    
    # a = boosting(ttbar2, pair1_t, pair1_tbar)
    # dic['boosted_cross_ttbar2_x'] = a[0]
    # dic['boosted_cross_ttbar2_y'] = a[1]
    
    # for i in range(1,3):
    #     pair1 = ttbars[:,i]
    #     pair2_t = t[:,1- i//2].boostCM_of(pair1)
    #     pair2_t_p = norm_p(pair2_t)
    #     pair2_tbar = tbar[:,1-i%2].boostCM_of(pair1)
    #     pair2_tbar_p = norm_p(pair2_tbar)
    #     beam_axis = vector.obj(x=0, y=0, z=1, t=0).boost(pair1)
    #     beam_axis_p = norm_p(beam_axis)
    #     a = np.cross(pair2_t_p, pair2_tbar_p, axis=0)
    #     dic[f'boosted_cross_all_{i}x'] = a[0]
    #     dic[f'boosted_cross_all_{i}y'] = a[1]
    #     dic[f'boosted_cross_{i}z'] = a[2]
    #     dic[f'CS_angle_t{i}'] = np.einsum('ij,ij->i', pair2_tbar_p.T, beam_axis_p.T)
    #     dic[f'CS_angle_tbar{i}'] = np.einsum('ij,ij->i', pair2_t_p.T, beam_axis_p.T)
    

    return dic

def make_all_data(path, filenames):
    all_data = {}
    for file in filenames:
        base = uproot.open(path+file)['tree']
        all_data[file.split('.root')[0]] = form_data(base)
        all_data[file.split('.root')[0]]['weights'] = base['weights'].array()
    return all_data


def find_run_info(path):
    full_info = {}
    name_next = True
    dic = None
    with open(path+'run_info.txt', 'r') as f:
        for line in f:
            if '-'*40 in line:
                name_next = True
                if dic:
                    dic['wilson'] = name.split('_')[1]
                    full_info[name] = dic
            elif name_next:
                dic = {}
                name = line.strip()
                name_next = False
            else:
                line = line.strip().split(' : ')
                dic[line[0]] = line[1]

    return full_info

def plot_wilson(dics, plotting, bins = 50,title = False):
    vars = list(list(dics.values())[0].keys())
    hep.style.use(hep.style.ATLAS)  # Set the style once at the beginning
    
    # plots all the variables
    for var in vars:
        hists = []
        for name, dic in dics.items():
            high_limit = np.max(np.concatenate([dic[var] for dic in dics.values()]))
            low_limit = min(0,np.min(np.concatenate([dic[var] for dic in dics.values()])))
            if plotting['lims'].get(var):
                hist = bh.Histogram(bh.axis.Regular(bins, plotting['lims'][var][0], plotting['lims'][var][1]))
            else:
                hist = bh.Histogram(bh.axis.Regular(bins, min(low_limit, 0), high_limit))
            hist.fill(dic[var])
            hists.append(hist)
            print(name)
            # print(f'{name}, hist_size: {hist}')
    
        plt.figure()  # Initialize a new figure for each plot
        # hep.histplot([hist/hist.sum() for hist in hists], label=[name for name in dics.keys()])
        hep.histplot([hist for hist in hists], label=[name for name in dics.keys()])
        if plotting['lims'].get(var):
            plt.ylim(plotting['lims'][var][2], plotting['lims'][var][3])
        if plotting['names'].get(var):
            name = plotting['names'][var]
            plt.xlabel(rf'${name}$')
        else:
            plt.xlabel(f'{var}')  
        plt.ylabel('Frequency')
        if title:
            plt.title(f'{var}')
        plt.legend()
        #plt.savefig(f'EFT_{var}_{plotting_names[variable]}.png', dpi=300)
        plt.show()
        
    
def plotting_vals():
    plotting = {}
    plotting['lims'] = {}
    plotting['lims']['Ht'] = [0, 3000, 0, 4500]
    plotting['lims']['M_tttt'] = [0, 3500, 0, 3500]
    plotting['lims']['max_eta_ttbar'] = [-6, 6, 0, 3500]
    plotting['lims']['max_pt'] = [0, 1400, 0, 4500]
    plotting['lims']['leading_pt_b4'] = [-1, 1, 0, 2000]


    plotting['names'] = {}
    plotting['names']['Ht'] = r'H_{T}'
    plotting['names']['M_tttt'] = r'M_{tttt}'
    plotting['names']['max_eta_ttbar'] = r'\eta_{\mathrm{max}}'
    plotting['names']['max_pt'] = r'p_{T\mathrm{max}}'
    plotting['names']['leading_pt_b4'] = r'b4[PT_{max}]'
    
    return plotting

def plot_variables(all_data, run_info, wilsons, bins, plotting={}):
    for wilson in wilsons:
        data = {'EFT_sm': all_data['EFT_sm']}
        values = [float(run_info[f'EFT_{wilson}_lower'][wilson]), float(run_info[f'EFT_{wilson}_higher'][wilson])]
        data[f'{wilson}=+{values[1]}'] = all_data[f'EFT_{wilson}_higher']
        data[f'{wilson}={values[0]}'] = all_data[f'EFT_{wilson}_lower']
        plot_wilson(data, plotting, bins = 10)
        
def quad_formula(y, a,b,c):
    part = b**2-4*a*(c-y)
    if part<0:
        return [0, 0]
    part2 = np.sqrt(part)
    return np.sort([(-b-part2)/(2*a), (-b+part2)/(2*a)])

quad = lambda x, a,b,c: a*x**2+b*x+c

def plot_quad(run_info,wilson, y_test=None):
    x = [float(run_info[f'EFT_{wilson}_lower'][wilson]), 0, float(run_info[f'EFT_{wilson}_higher'][wilson])]
    y = [float(run_info[f'EFT_{wilson}_lower']['cross_section']), float(run_info['EFT_sm']['cross_section']), float(run_info[f'EFT_{wilson}_higher']['cross_section'])]
    a, b, c = np.polyfit(x, y, 2)
    vals = [x[0], x[2]]
    
    if y_test is not None:
        result = quad_formula(y_test, a,b,c)
        if result is not None:
            vals = [min(result[0], x[0]), max(result[1], x[2])]
            
    x2 = np.linspace(vals[0]-1, vals[1]+1, 100)
    quad = lambda x: a*x**2+b*x+c
    plt.plot(x2, quad(x2))
    plt.plot(x,y, 'kx')
    if result is not None:
        plt.plot(result, [y_test,y_test], 'rx')
    plt.xlabel(wilson)
    plt.ylabel(r'$\sigma$', loc='center')
    plt.show()
    return [a,b,c]

def find_coefs(all_data, vars, run_info, wilson, plotting, bins):
        # inputs an array of target hists, with an array of strings for the variables used (must be in order)
    coefs = {}
    for var in vars:
        plot_info = plotting['lims'][var]
        hist1 = bh.Histogram(bh.axis.Regular(bins,plot_info[0] , plot_info[1]))
        hist1.fill(all_data[f'EFT_{wilson}_lower'][var])
        hist2 = bh.Histogram(bh.axis.Regular(bins,plot_info[0] , plot_info[1]))
        hist2.fill(all_data['EFT_sm'][var])
        hist3 = bh.Histogram(bh.axis.Regular(bins,plot_info[0] , plot_info[1]))
        hist3.fill(all_data[f'EFT_{wilson}_higher'][var])
        x = [float(run_info[f'EFT_{wilson}_lower'][wilson]), 0, float(run_info[f'EFT_{wilson}_higher'][wilson])]

        coef = np.empty((0,3))
        for i in range(bins):
            y = np.array([hist1[i], hist2[i], hist3[i]])#*cross_sections
            coef = np.vstack((coef, np.polyfit(x, y, 2)))
        coefs[var] = coef
    return coefs

def compare_hists(target_hists, vars, global_coefs, wilson, wilson_value,plotting, bins):
    for hist, var in zip(target_hists,vars):
        plot_info = plotting['lims'][var]
        coefs = global_coefs[var]
        hist_make = bh.Histogram(bh.axis.Regular(bins,plot_info[0] , plot_info[1]))
        for i in range(bins):
            hist_make[i] = quad(wilson_value, *coefs[i])
            
        hep.histplot([hist, hist_make], label=['custom', f'custom: {wilson}={wilson_value:.4f}'])
        name = plotting['names'][var]
        plt.xlabel(rf'${name}$')
        plt.legend()
        plt.show()
        
def form_hists(all_data, vars, global_coefs, run_info, wilson_value, bins, plotting, noise=False): 
    hists = []
    for var in vars:
        plot_info = plotting['lims'][var]
        hist1 = bh.Histogram(bh.axis.Regular(bins,plot_info[0] , plot_info[1]))
        hist1.fill(all_data['EFT_cQQ1_lower'][var])
        hist2 = bh.Histogram(bh.axis.Regular(bins,plot_info[0] , plot_info[1]))
        hist2.fill(all_data['EFT_sm'][var])
        hist3 = bh.Histogram(bh.axis.Regular(bins,plot_info[0] , plot_info[1]))
        hist3.fill(all_data['EFT_cQQ1_higher'][var])
        coefs = global_coefs[var]
    
        hist_make = bh.Histogram(bh.axis.Regular(bins,plot_info[0] , plot_info[1]))
        for i in range(bins):
            hist_make[i] = quad(wilson_value, *coefs[i])
            if noise:
                hist_make[i] = np.random.poisson(hist_make[i])
        
        hists.append(hist_make)
        lower_val = float(run_info[f'EFT_cQQ1_lower']['cQQ1'])
        higher_val = float(run_info[f'EFT_cQQ1_higher']['cQQ1'])
        hep.histplot([hist1, hist2, hist3, hist_make], label=[f'lower: {lower_val}', 'sm', f'higher: {higher_val}', f'custom: {wilson_value}'])
        name = plotting['names'][var]
        plt.xlabel(rf'${name}$')
        plt.legend()
        plt.show()
        
    return hists

def general(target_hists, vars, global_coefs, wilson, bins, plotting):
    # inputs an array of target hists, with an array of strings for the variables used (must be in order)
    coefs = np.concatenate([global_coefs[var] for var in vars])

    data = np.concatenate([hist.values() for hist in target_hists])
    mask = data!=0
    data = data[mask]
    coefs = coefs[mask]
    
    roots = np.empty((0,2))
    for value, coef in zip(data, coefs):
        roots = np.vstack((roots, quad_formula(value, *coef)))
    initial_guess_min = np.mean(roots[:,0])
    initial_guess_max = np.mean(roots[:,1])
    
    def form_chi(coef):
        # this is actually including errors since the error is sqrt(target)
        return np.sum([(quad(coef, *fitting)-target)**2/target for fitting, target in zip(coefs, data)], axis=0)
    
    results_min = scipy.optimize.minimize(form_chi, initial_guess_min, bounds = [[-10,10]])
    results_max = scipy.optimize.minimize(form_chi, initial_guess_max, bounds = [[-10,10]])
    arg = np.argmin([results_min['fun'], results_max['fun']])
    value = [results_min['x'], results_max['x']][arg][0]
    chi = [results_min['fun'], results_max['fun']][arg]
    hess = [results_min['hess_inv'].todense()[0][0], results_max['hess_inv'].todense()[0][0]][arg]
    print('value of coefficient found as {0:.5f}+- {1:.5f} with a chi squared of {2:.3f}'.format(value,np.sqrt(hess), chi))
    
    
    uncer_min_results = scipy.optimize.minimize(lambda x :(chi+1 - form_chi(x))**2, value, bounds = [[-15,value-0.000001]])
    uncer_max_results = scipy.optimize.minimize(lambda x :(chi+1 - form_chi(x))**2, value, bounds = [[value+0.000001,1]])
    uncer_min1 = value- uncer_min_results['x'][0]
    uncer_max1 = uncer_max_results['x'][0]-value

    print('value of coefficient found as {0:.5f} +{1:.5f} -{2:.5f} with a chi squared of {3:.3f}'.format(value,uncer_max1, uncer_min1, chi))
    
    uncer_min_results = scipy.optimize.minimize(lambda x :(chi+4 - form_chi(x))**2, value, bounds = [[-15,value-0.000001]])
    uncer_max_results = scipy.optimize.minimize(lambda x :(chi+4 - form_chi(x))**2, value, bounds = [[value+0.000001,1]])
    uncer_min2 = value- uncer_min_results['x'][0]
    uncer_max2 = uncer_max_results['x'][0]-value

    print('value of coefficient found as {0:.5f} +{1:.5f} -{2:.5f} (95%) with a chi squared of {3:.3f}'.format(value,uncer_max2, uncer_min2, chi))
    
    compare_hists(target_hists, vars, global_coefs, wilson, value,plotting, bins)
    return value, [[uncer_min1, uncer_min1], [uncer_min2, uncer_min2]]