import pickle
nLabel, nTrial, nUser, nChannel, nTime = 4, 40, 32, 40, 8064
nUsers = 32

def dataconvert():
    print("----------Data Convert----------")
    fout_data = open("data/processed/features_raw.dat", 'w')
    fout_labels0 = open("data/processed/labels_0.dat", 'w')
    fout_labels1 = open("data/processed/labels_1.dat", 'w')
    fout_labels2 = open("data/processed/labels_2.dat", 'w')
    fout_labels3 = open("data/processed/labels_3.dat", 'w')
    for i in range(nUsers):
        if i < 10:
            filenumber = '%0*d' % (2, i + 1)
        else:
            filenumber = i + 1
        filename = "data/s" + str(filenumber) + ".dat"
        f = open(filename, 'rb')  # Read the file in Binary mode
        x = pickle.load(f, encoding='latin1')
        print(filename)

        for tr in range(nTrial):
            for dat in range(nTime):
                for ch in range(nChannel):
                    fout_data.write(str(x['data'][tr][ch][dat]) + " ");               # data (32*40)*(40*8064)  (user*trial)*(channel*time)
            fout_labels0.write(str(x['labels'][tr][0]) + "\n");                       # label (32*40) user*trial
            fout_labels1.write(str(x['labels'][tr][1]) + "\n");
            fout_labels2.write(str(x['labels'][tr][2]) + "\n");
            fout_labels3.write(str(x['labels'][tr][3]) + "\n");
            fout_data.write("\n");
    fout_labels0.close()
    fout_labels1.close()
    fout_labels2.close()
    fout_labels3.close()
    fout_data.close()
    print("--------------------" + "\n")


if __name__ == '__main__':
    dataconvert()
