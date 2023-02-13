- Many parts of the code are adapted from [seam-carving](https://github.com/andrewdcampbell/seam-carving)
- Reference: https://github.com/esimov/caire

# Paper Summary
- [Seam Carving for Content-Aware Image Resizing](https://perso.crans.org/frenoy/matlab2012/seamcarving.pdf)
- Effective resizing of images should not only use geometric constraints, but consider the image content as well. We present a simple image operator called seam carving that supports content-aware image resizing for both reduction and expansion.
- *A seam is an optimal 8-connected path of pixels on a single image from top to bottom, or left to right, where optimality is defined by an image energy function.* By repeatedly carving out or inserting seams in one direction we can change the aspect ratio of an image. By applying these operators in both directions we can retarget the image to a new size.
- *The selection and order of seams protect the content of the image, as defined by the energy function. Seam carving can also be used for image content enhancement and object removal. We support various visual saliency measures for defining the energy of an image, and can also include user input to guide the process. By storing the order of seams in an image we create multi-size images, that are able to continuously change in real time to fit a given size.*
- *Seam carving uses an energy function defining the importance of pixels. A seam is a connected path of low energy pixels crossing the image from top to bottom, or from left to right.* By successively removing or inserting seams we can reduce, as well as enlarge, the size of an image in both directions.
- *For image reduction, seam selection ensures that while preserving the image structure, we remove more of the low energy pixels and fewer of the high energy ones. For image enlarging, the order of seam insertion ensures a balance between the original image content and the artificially inserted pixels.*
- We illustrate the application of seam carving and insertion for aspect ratio change, image retargeting, image content enhancement, and object removal. *Furthermore, by storing the order of seam removal and insertion operations, and carefully interleaving seams in both vertical and horizontal directions we define multi-size images. Such images can continuously change their size in a content-aware manner. A designer can author a multi-size image once, and the client application, depending on the size needed, can resize the image in real time to fit the exact layout or the display.*
- *Seam carving can support several types of energy functions such as gradient magnitude, entropy, visual saliency, eye-gaze movement, and more. The removal or insertion processes are parameter free; however, to allow interactive control, we also provide a scribble-based user interface for adding weights to the energy of an image and guide the desired results.*
## Related Works
- *Top down methods use tools such as face detectors to detect important regions in the image, whereas bottom-up methods rely on visual saliency methods to construct a visual saliency map of the image. Once the saliency map is constructed, cropping can be used to display the most important region of the image.*
## Methology
- Intuitively, our goal is to remove unnoticeable pixels that blend with their surroundings.

# Paper Summary
- [Improved Seam Carving for Video Retargeting](http://www.eng.tau.ac.il/~avidan/papers/vidret.pdf)