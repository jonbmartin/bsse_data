import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg

figa = mpimg.imread('figures_out/rfmotion_pulse.png')
figb = mpimg.imread('figures_out/rfmotion_nonresonant.png')
figc = mpimg.imread('figures_out/rfmotion_resonant.png')

pyplot.figure()
pyplot.subplot(131)
pyplot.imshow(figa)
pyplot.axis('off')
pyplot.subplot(132)
pyplot.imshow(figb)
pyplot.axis('off')
pyplot.subplot(133)
pyplot.imshow(figc)
pyplot.axis('off')
pyplot.tight_layout()

pyplot.subplots_adjust(left=0.125,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.01,
                    hspace=0.01)

pyplot.show()
