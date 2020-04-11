import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

def plot_confusion_matrix(confidence_matrix, fileName):
    activity = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

    fig, ax = plt.subplots()
    im = ax.imshow(confidence_matrix)

    ax.set_xticks(np.arange(len(activity)))
    ax.set_yticks(np.arange(len(activity)))
    ax.set_xticklabels(activity)
    ax.set_yticklabels(activity)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(activity)):
        for j in range(len(activity)):
            text = ax.text(j, i, confidence_matrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusion matrix")
    fig.tight_layout()
    plt.show()
    if fileName!=None:
        fig.savefig(fileName)
    return

def plot_per_activity_metric(precision, recall, fmeasure, fileName):
    activities = 6
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(activities)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, precision, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Precision')

    rects2 = plt.bar(index + bar_width, recall, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Recall')

    rects3 = plt.bar(index + 2 * bar_width, fmeasure, bar_width,
                     alpha=opacity,
                     color='r',
                     label='F-measure')

    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Score value')
    plt.title('Evaluation by activities')
    plt.xticks(index + bar_width, ('Walking', 'WU', 'WD', 'SITTING', 'STANDING', 'LAYING'))
    plt.legend()

    plt.tight_layout()
    plt.show()
    if fileName!=None:
        fig.savefig(fileName)
    return

if __name__ == '__main__':
    # d = {1:	0.8916112956810631, 2:	0.9252873563218391, 3:	0.9834254143646409, 4:	0.9621451104100947, 5:	0.9006622516556292, 6:	0.8153846153846154, 7:	0.948051948051948, 8:	0.8612099644128114, 9:	0.8819444444444444, 10:	0.8979591836734694, 11: 0.9936708860759493, 12: 0.9375, 13: 0.963302752293578, 14: 0.631578947368421, 15: 0.9908536585365854, 16: 0.855191256830601, 17: 0.9619565217391305, 18: 0.9258241758241759, 19: 0.8694444444444445, 20: 0.9152542372881356, 21: 0.9019607843137255, 22: 0.9844236760124611, 23: 0.9489247311827957, 24: 0.968503937007874, 25: 0.8533007334963325, 26: 0.9132653061224489, 27: 0.9840425531914894, 28: 0.918848167539267, 29: 0.9680232558139535, 30: 0.9817232375979112}
    # d = {1: 0.9712105727362579, 2: 0.9356528985646669, 3: 0.9733881442741805, 4: 0.9493817187723109, 5: 0.8574320184600653, 6: 0.8576793571540021, 7: 0.9545454545454546, 8: 0.9026238854383705, 9: 0.8835794558248327, 10: 0.8691074335520018, 11: 0.9905107914739175, 12: 0.9812501174216093, 13: 0.962492179657267, 14: 0.718746482945394, 15: 0.9939071348741981, 16: 0.8736903260673753, 17: 0.9727270927377841, 18: 0.8413317225089902, 19: 0.8938806850686499, 20: 0.9633615905193202, 21: 0.9321172702719621, 22: 0.9846007574979538, 23: 0.9648236412844707, 24: 0.9973745048512079, 25: 0.8061553283586259, 26: 0.994900738073007, 27: 0.9867111190073637, 28: 0.8961790252808974, 29: 0.9592125759069281, 30: 0.9817014790141861}
    d = {1: 0.9016121566553844, 2: 0.8422408817786547, 3: 0.8780004924230215, 4: 0.9235167572015359, 5: 0.9145356725807838, 6: 0.8483101198078007, 7: 0.8589746388065717, 8: 0.7922432850658053, 9: 0.7874276388262129, 10: 0.834942877760093, 11: 0.6607078066195048, 12: 0.8213042268189327, 13: 0.8299498364904285, 14: 0.6474619648518669, 15: 0.7986301064483955, 16: 0.8134218336959761, 17: 0.8472389431659594, 18: 0.7151780847530901, 19: 0.8377842968816213, 20: 0.8579061578817303, 21: 0.8963605535096245, 22: 0.9091823141667378, 23: 0.8555848273349699, 24: 0.8992949461909335, 25: 0.7261350670033382, 26: 0.9464058874615993, 27: 0.9573531276127806, 28: 0.8293407783665474, 29: 0.8490149303712156, 30: 0.8844926635552501}
    lis = []
    for key, value in d.items():
        lis.append((key, value))
    lis = sorted(lis, key=lambda x: x[1])
    objects = [i[0] for i in lis]
    performance = [i[1] for i in lis]
    y_pos = np.arange(len(objects))

    # performance = list(d.values())

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Performance score')
    plt.title('Subject cross validation for k-Nearest Neighbours')
    plt.savefig('k-NN subject cross validation.png')
    plt.show()

    # a = np.array([[482,   3,  11,   0,   0,   0],
    #          [ 42, 423,   6,   0,   0,   0],
    #          [ 48,  40, 332,   0,   0,   0],
    #          [  0,   4,   0, 394,  93,   0],
    #          [  0,   0,   0,  35, 497,   0],
    #          [  0,   0,   0,   2,   1, 534]])
    # plot_confusion_matrix(a, "test.png")
