

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b a)
(ontable c)
(on d c)
(clear b)
(clear d)
)
(:goal
	(and
		(on b d)
		(on c a)
		(on d c)
		(on c d)
	)
)
)