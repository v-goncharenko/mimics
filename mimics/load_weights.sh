# Downloads and puts to appropriate place weights of dlib model

echo 'Started downloading dlib model'
mkdir dlib-models
cd dlib-models
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
echo 'Successfully downloaded dlib model!'
