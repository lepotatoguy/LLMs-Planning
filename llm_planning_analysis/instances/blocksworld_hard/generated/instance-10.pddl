

(define (problem BW-rand-13)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j k l m )
(:init
(handempty)
(on a e)
(on b l)
(ontable c)
(on d g)
(on e f)
(ontable f)
(on g j)
(on h m)
(ontable i)
(ontable j)
(on k h)
(on l k)
(on m i)
(clear a)
(clear b)
(clear c)
(clear d)
)
(:goal
(and
(on a j)
(on b h)
(on c d)
(on d f)
(on e i)
(on f b)
(on g k)
(on h l)
(on i a)
(on j c)
(on l g)
(on m e))
)
)


