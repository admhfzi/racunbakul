# PINENG 22.5W powerbank: hand tilt video (9:16)

- `pineng_hand_tilt_9x16.mp4`: 1080x1920, 30fps, 8s. A hand in a yellow sweatshirt holds the powerbank and tilts it very slowly (about ±2°). The product stays in place.
- The product pixels come straight from `assets/product_source.jpg`. They were cut out (`cut_product.py`) without being redrawn, so the label and details are unchanged.
- Background: the original desk photo, cropped from the sides to fit 9:16. Nothing was added to it or changed.
- The hand with the yellow sleeve was AI-generated in Canva and then cut out (`assets/hand.png`).

Re-render: `cd pineng-powerbank && python3 make_video.py out.mp4 1.1 1.0 -30 10 200`
(args: product scale, hand scale, hand rotation, hand x/y offset)
