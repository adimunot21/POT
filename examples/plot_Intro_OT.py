# This code is an example of bakeries sending croissants to cafes in NYC
# There is a finite number of bakeries, a finite number of cafes, with a finite number of croissants in each bakery, and required by the cafes 

import numpy as np  
import pylab as pl 
import ot
import time

##############################################################################
# First OT Problem
# ----------------
#
# The dataset has a picture of a google map snap with locations of bakeries and cafes
# It also has a fictional supply and demand values of croissants assigned to each bakery/cafe

# We have access to the position of Bakeries ``bakery_pos`` and their
# respective production ``bakery_prod`` which describe the source
# distribution. 

# The Cafés where the croissants are sold are defined also by
# their position ``cafe_pos`` and ``cafe_prod``, and describe the target
# distribution. 
# ``Imap`` will illustrate the position of these shops in the city.
###############################################################################

# Load the data and assign variables

data = np.load(r'C:\Users\adimu\OneDrive\Documents\Imperial Tings\Year 4\Masters\Python OT Experimentation\POT\data\manhattan.npz')

bakery_pos = data['bakery_pos']
bakery_prod = data['bakery_prod']
cafe_pos = data['cafe_pos']
cafe_prod = data['cafe_prod']
Imap = data['Imap']

print('Bakery production: {}'.format(bakery_prod))
print('Cafe sale: {}'.format(cafe_prod))
print('Total croissants : {}'.format(cafe_prod.sum()))




##############################################################################
# Plot the positions of bakeries and cafes. 
# The size of the circle is proportional to their demand or supply
##############################################################################

pl.figure(1, (7, 6))
pl.clf()
pl.imshow(Imap, interpolation='bilinear')  # plot the map
pl.scatter(bakery_pos[:, 0], bakery_pos[:, 1], s=bakery_prod, c='r', ec='k', label='Bakeries')
pl.scatter(cafe_pos[:, 0], cafe_pos[:, 1], s=cafe_prod, c='b', ec='k', label='Cafés')
pl.legend()
pl.title('Manhattan Bakeries and Cafés')
# pl.show()





# ##############################################################################
# Cost Matrix

# The cost matrix can be computed by using 'ot.dist' function
# This finds the default squared eucladian distance between the bakeries and the cafes 
# ##############################################################################

C = ot.dist(bakery_pos, cafe_pos)

labels = [str(i) for i in range(len(bakery_prod))]
f = pl.figure(2, (14, 7))
pl.clf()
pl.subplot(121)
pl.imshow(Imap, interpolation='bilinear')  # plot the map
for i in range(len(cafe_pos)):
    pl.text(cafe_pos[i, 0], cafe_pos[i, 1], labels[i], color='b',
            fontsize=14, fontweight='bold', ha='center', va='center')
for i in range(len(bakery_pos)):
    pl.text(bakery_pos[i, 0], bakery_pos[i, 1], labels[i], color='r',
            fontsize=14, fontweight='bold', ha='center', va='center')
pl.title('Manhattan Bakeries and Cafés')

# A colorbar is added next to the image to display the cost matrix
# The columns are the bakeries and rows are the cafes 
# The red cells in the matrix image show the bakeries and cafés that are
# further away, and thus more costly to transport from one to the other, while
# the blue ones show those that are very close to each other, with respect to
# the squared Euclidean distance.

ax = pl.subplot(122)
im = pl.imshow(C, cmap="coolwarm")
pl.title('Cost matrix')
cbar = pl.colorbar(im, ax=ax, shrink=0.5, use_gridspec=True)
cbar.ax.set_ylabel("cost", rotation=-90, va="bottom")
pl.xlabel('Cafés')
pl.ylabel('Bakeries')
pl.tight_layout()
# pl.show()






##############################################################################
# Solving the OT problem with 'ot.emd'
# This function returns the transport matrix. 
# We are also logging the time it took to perform the computation
##############################################################################

start = time.time()
ot_emd = ot.emd(bakery_prod, cafe_prod, C)
time_emd = time.time() - start

print(time_emd)





##############################################################################
# Transportation plan visualization

# We will plot the matrix 



# # Plot the matrix and the map
f = pl.figure(3, (14, 7))
pl.clf()
pl.subplot(121)
pl.imshow(Imap, interpolation='bilinear')

#Plot the lines from bakeries to cafe with their width according to the values obtained from ot.emd
for i in range(len(bakery_pos)):
    for j in range(len(cafe_pos)):
        pl.plot([bakery_pos[i, 0], cafe_pos[j, 0]], [bakery_pos[i, 1], cafe_pos[j, 1]],
                '-k', lw=3. * ot_emd[i, j] / ot_emd.max())
        
#Label the bakeries and cafes
for i in range(len(cafe_pos)):
    pl.text(cafe_pos[i, 0], cafe_pos[i, 1], labels[i], color='b', fontsize=14,
            fontweight='bold', ha='center', va='center')
for i in range(len(bakery_pos)):
    pl.text(bakery_pos[i, 0], bakery_pos[i, 1], labels[i], color='r', fontsize=14,
            fontweight='bold', ha='center', va='center')
pl.title('Manhattan Bakeries and Cafés')

ax = pl.subplot(122)
im = pl.imshow(ot_emd)
for i in range(len(bakery_prod)):
    for j in range(len(cafe_prod)):
        text = ax.text(j, i, '{0:g}'.format(ot_emd[i, j]),
                       ha="center", va="center", color="w")
pl.title('Transport matrix')

pl.xlabel('Cafés')
pl.ylabel('Bakeries')
pl.tight_layout()
# pl.show()






##############################################################################
# The final wasserstein loss, or the expected cost is this transport plan multiplied by the cost matrix
##############################################################################

W = np.sum(ot_emd * C)
print('Wasserstein loss (EMD) = {0:.2f}'.format(W))



##############################################################################
# Regularized OT with Sinkhorn
#
# The Sinkhorn algorithm is very simple to code. You can implement it directly
# using the pseudo-code
#
# An alternative is to use the POT toolbox with 'ot.sinkhorn'
#
# Be careful of numerical problems. A good pre-processing for Sinkhorn is to
# divide the cost matrix ``C`` by its maximum value.

##############################################################################
#Set the regularization parameter
reg = 0.1


##############################################################################
# Algorithm hard code
##############################################################################

# K = np.exp(-C / C.max() / reg)
# nit = 100
# u = np.ones((len(bakery_prod), ))
# for i in range(1, nit):
#     v = cafe_prod / np.dot(K.T, u)
#     u = bakery_prod / (np.dot(K, v))
# ot_sink_algo = np.atleast_2d(u).T * (K * v.T)  # Equivalent to np.dot(np.diag(u), np.dot(K, np.diag(v)))
# print(ot_sink_algo)



##############################################################################
# Sinkhorn using ot.sinkhorn
##############################################################################
start_sink = time.time()
ot_sinkhorn = ot.sinkhorn(bakery_prod, cafe_prod, reg=reg, M=C / C.max())
# print(ot_sinkhorn)
time_sink = time.time() - start

print(time_sink)


##############################################################################
# Plot the matrix and the map
##############################################################################

print('Min. of Sinkhorn\'s transport matrix = {0:.2g}'.format(np.min(ot_sinkhorn)))

f = pl.figure(4, (13, 6))
pl.clf()
pl.subplot(121)
pl.imshow(Imap, interpolation='bilinear')  # plot the map
for i in range(len(bakery_pos)):
    for j in range(len(cafe_pos)):
        pl.plot([bakery_pos[i, 0], cafe_pos[j, 0]],
                [bakery_pos[i, 1], cafe_pos[j, 1]],
                '-k', lw=3. * ot_sinkhorn[i, j] / ot_sinkhorn.max())
for i in range(len(cafe_pos)):
    pl.text(cafe_pos[i, 0], cafe_pos[i, 1], labels[i], color='b',
            fontsize=14, fontweight='bold', ha='center', va='center')
for i in range(len(bakery_pos)):
    pl.text(bakery_pos[i, 0], bakery_pos[i, 1], labels[i], color='r',
            fontsize=14, fontweight='bold', ha='center', va='center')
pl.title('Manhattan Bakeries and Cafés')

ax = pl.subplot(122)
im = pl.imshow(ot_sinkhorn)
for i in range(len(bakery_prod)):
    for j in range(len(cafe_prod)):
        text = ax.text(j, i, np.round(ot_sinkhorn[i, j], 1),
                       ha="center", va="center", color="w")
pl.title('Transport matrix')

pl.xlabel('Cafés')
pl.ylabel('Bakeries')
pl.tight_layout()
# pl.show()



##############################################################################
# We notice right away that the matrix is not sparse at all with Sinkhorn,
# each bakery delivering croissants to all 5 cafés with that solution. Also,
# this solution gives a transport with fractions, which does not make sense
# in the case of croissants. This was not the case with EMD.
##############################################################################


##############################################################################
# Varying the regularization parameter in Sinkhorn
##############################################################################

reg_parameter = np.logspace(-3, 0, 20)
W_sinkhorn_reg = np.zeros((len(reg_parameter), ))
time_sinkhorn_reg = np.zeros((len(reg_parameter), ))

f = pl.figure(5, (14, 5))
pl.clf()
max_ot = 100  # plot matrices with the same colorbar
for k in range(len(reg_parameter)):
    start = time.time()
    ot_sinkhorn = ot.sinkhorn(bakery_prod, cafe_prod, reg=reg_parameter[k], M=C / C.max())
    time_sinkhorn_reg[k] = time.time() - start

    if k % 4 == 0 and k > 0:  # we only plot a few
        ax = pl.subplot(1, 5, k // 4)
        im = pl.imshow(ot_sinkhorn, vmin=0, vmax=max_ot)
        pl.title('reg={0:.2g}'.format(reg_parameter[k]))
        pl.xlabel('Cafés')
        pl.ylabel('Bakeries')

    # Compute the Wasserstein loss for Sinkhorn, and compare with EMD
    W_sinkhorn_reg[k] = np.sum(ot_sinkhorn * C)
pl.tight_layout()
# pl.show()

##############################################################################
# This series of graph shows that the solution of Sinkhorn starts with something
# very similar to EMD (although not sparse) for very small values of the
# regularization parameter, and tends to a more uniform solution as the
# regularization parameter increases.
#

##############################################################################
# Wasserstein loss and computational time
# ```````````````````````````````````````
#

# Plot the matrix and the map
f = pl.figure(6, (4, 4))
pl.clf()
pl.title("Comparison between Sinkhorn and EMD")

pl.plot(reg_parameter, W_sinkhorn_reg, 'o', label="Sinkhorn")
XLim = pl.xlim()
pl.plot(XLim, [W, W], '--k', label="EMD")
pl.legend()
pl.xlabel("reg")
pl.ylabel("Wasserstein loss")
# pl.show()
