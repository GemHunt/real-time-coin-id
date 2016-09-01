Select imageID
from images
where imageID % 2 = 0
and x_hough is not null
and imageID not in (select imageID from images where bad_image = 1)
and imageID not in (select imageID - 1 from images where bad_image = 1 and imageID % 2 = 1)
and imageID not in (select imageID from images where heads is not null )
and imageID not in (select imageID - 1 from images where heads is not null and imageID % 2 = 1)
Order by 1