from gpaw.tddft.spectrum import photoabsorption_spectrum

# Calculate spectrum
photoabsorption_spectrum('dm-length_x.dat', 'spec.dat', width=0.1, delta_e=0.01)
