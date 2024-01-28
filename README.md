Ray Tracing in One Weekend in CUDA using a wavefront algorithm
==================================

The original raytracing in one weekend:

By Roger Allen
May, 2018

See the [Master Branch](https://github.com/rogerallen/raytracinginoneweekend) for more information.

Th original Raytracing in one weekend in cuda:

See the [Master Branch](https://github.com/rogerallen/raytracinginoneweekendincuda) for more information.


our version implements a wavefront algorithm on top of these previous codebases.


How to run:
Our implementation runs on windows 10/11, if you are running on Linux, you can use the Makefile in the original project repos. To run the project on windows, edit the .mk file so the CUDA\_PATH correctly corresponds to your CUDA install location and use GENCODE\_FLAGS that are supported by your graphics card. To run the project, you can use the following command "make -f {name}.mk out.png". To run this command you require a version of make (We used "Make for Windows") and ImageMagick to convert the .ppm file to a .jpg.   

On line 265 of main.cu a boolean can be toggled to either use the wavefront algorithm or the mega kernel algorithm.  Variable ns on line 292 of main.cu can be changed to change the number of rays per pixel.
