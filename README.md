# Computer graphics homework - BMEVIIIAB07

# First homework
Create a program that displays a random graph aesthetically and allows the user to zoom in on any part of it while the rest of the graph is still visible. The graph consists of 50 nodes with a saturation of 5% (5% of possible edges are actual edges). For aesthetic layout, the position of the nodes must be determined by heuristic and force-directed graph drawing algorithms based on the rules of the hyperbolic plane when the SPACE key is pressed.

To focus on specific parts of the graph, it should be arranged on the hyperbolic plane and projected onto the screen using the Beltrami-Klein method. To focus, the graph is shifted on the hyperboloid so that the interesting part is at the bottom. The shift is triggered by pressing the right mouse button and dragging the mouse while it is in the pressed state.

The individual nodes are represented by circles on the hyperbolic plane, which are textured with the identifier of the corresponding node.

# Second homework
Create a ray tracing program that displays a room in the shape of a dodecahedron that can be inscribed in a sphere with a radius of √3 meters. The room contains a smooth optical gold object with a 0.3 meter radius, located at the center of the room, defined by an implicit equation 𝑓(𝑥,𝑦,𝑧)=exp⁡(𝑎𝑥^2+𝑏𝑦^2−𝑐𝑧)−1, and a point light source. The walls of the room are of diffuse-specular type from the corners up to 0.1 meters, inside which there are portals to other similar rooms rotated by 72 degrees around the center of the wall and mirrored to the plane of the wall. The light source does not shine through the portal, and each room has its own light source. During the rendering, it is sufficient to pass through the portals a maximum of 5 times. The virtual camera looks at the center of the room and rotates around it. The refractive index and extinction coefficient of gold are: n/k: 0.17/3.1, 0.35/2.7, 1.5/1.9. The other parameters can be individually selected to create a beautiful image. 𝑎,𝑏,𝑐 are positive, non-integer numbers.
# Third homework
Gravity demonstration using a rubber mat simulator. We start by observing the flat torus topology (which goes out on one side and comes in on the opposite side) rubber mat from above. We can place large, non-moving masses on it by pressing the right mouse button, and slide small balls from the bottom left corner to the right using the left mouse button, where the location of the button press determines the initial velocity in conjunction with the bottom left corner. The stationary massive objects bend the space, i.e., deform the rubber mat, but they are not visible. The caused indentation at a distance 𝑟 from the center of mass is 𝑚/(𝑟 + 𝑟0), where 𝑟0 is half percent of the width of the rubber mat, and 𝑚 is the increasing mass of the placed objects. The rubber mat is optically rough, with the darkness gradually increasing according to the depth of the indentation and ambient factors. The balls are colored diffuse-specular objects, and their size and bending effects can be neglected. Pressing SPACE key, our virtual camera will follow the first ball that has not been absorbed yet. The balls that collide with the masses are absorbed, and we do not need to worry about collisions between the balls. Two point light sources illuminate the rubber mat, and they rotate around their initial position according to the following quaternion (where t is time): 𝑞=[𝑐𝑜𝑠(𝑡/4), 𝑠𝑖𝑛(𝑡/4)𝑐𝑜𝑠(𝑡)/2, 𝑠𝑖𝑛(𝑡/4)𝑠𝑖𝑛(𝑡)/2, 𝑠𝑖𝑛(𝑡/4)√(3/4])