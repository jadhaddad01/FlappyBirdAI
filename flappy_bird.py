import pygame
import neat
import time
import os
import random
import pygame_menu
from utils import UI, visualize, confmodif
import pickle
from PIL import Image
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800

win = UI.Window(WIN_WIDTH, WIN_HEIGHT)
button2 = UI.Button(
    x = 500 - 10 - 120,
    y = 50,
    w = 120,
    h = 35,
    param_options={
        'curve': 0.3,
        'text' : "Menu",
        'font_colour': (255, 255, 255),
        'background_color' : (200, 200, 200), 
        'hover_background_color' : (160, 160, 160),
        'outline_half': False
    }
)

gen = 0

neural_net_image = None

hs_genopt_popopt = [0, 5, 5] # Default
with open(os.path.join("utils", "hs_genopt_popopt.txt"), "rb") as fp: # Unpickling
    hs_genopt_popopt = pickle.load(fp)

BIRD_IMGS1 = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "1bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "1bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "1bird3.png")))]
BIRD_IMGS2 = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "2bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "2bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "2bird3.png")))]
BIRD_IMGS3 = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "3bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "3bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "3bird3.png")))]
BIRD_IMGS = [BIRD_IMGS1, BIRD_IMGS2, BIRD_IMGS3]


PIPE_IMG1 = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe1.png")))
PIPE_IMG2 = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe2.png")))
PIPE_IMGS = [PIPE_IMG1, PIPE_IMG2]

BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans", 50)
STAT_FONT_BIG = pygame.font.SysFont("comicsans", 100)

class Bird:
	random_color = random.randrange(1, 4) - 1
	IMGS = BIRD_IMGS[random_color]
	MAX_ROTATION = 25
	ROT_VEL = 20
	ANIMATION_TIME = 5

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.tilt = 0
		self.tick_count = 0
		self.vel = 0
		self.height = self.y
		self.img_count = 0

		self.random_color = random.randrange(1, 4) - 1
		self.IMGS = BIRD_IMGS[self.random_color]
		self.img = self.IMGS[0]

	def jump(self):
		self.vel = -10.5
		self.tick_count = 0
		self.height = self.y

	def move(self):
		self.tick_count += 1

		d = self.vel*self.tick_count + 1.5*self.tick_count**2

		if d >= 16:
			d = 16

		if d < 0:
			d -= 2

		self.y = self.y + d

		if d < 0 or self.y < self.height + 50:
			if self.tilt < self.MAX_ROTATION:
				self.tilt = self.MAX_ROTATION

		else:
			if self.tilt > -90:
				self.tilt -= self.ROT_VEL

	def draw(self, win):
		self.img_count += 1

		if self.img_count < self.ANIMATION_TIME:
			self.img = self.IMGS[0]
		elif self.img_count < self.ANIMATION_TIME*2:
			self.img = self.IMGS[1]
		elif self.img_count < self.ANIMATION_TIME*3:
			self.img = self.IMGS[2]
		elif self.img_count < self.ANIMATION_TIME*4:
			self.img = self.IMGS[1]
		elif self.img_count == self.ANIMATION_TIME*4 + 1:
			self.img = self.IMGS[0]
			self.img_count = 0

		if self.tilt <= -80:
			self.img = self.IMGS[1]
			self.img_count = self.ANIMATION_TIME*2

		rotated_image = pygame.transform.rotate(self.img, self.tilt)
		new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
		win.blit(rotated_image, new_rect.topleft)

	def get_mask(self):
		return pygame.mask.from_surface(self.img)

class Pipe:
	GAP = 200
	VEL = 5

	def __init__(self, x):
		self.x = x
		self.height = 0
		self.gap = 100

		self.top = 0
		self.bottom = 0

		random_color = random.randrange(1, 3) - 1
		self.PIPE_TOP = pygame.transform.flip(PIPE_IMGS[random_color], False, True)
		self.PIPE_BOTTOM = PIPE_IMGS[random_color]

		self.passed = False
		self.set_height()

	def set_height(self):
		self.height = random.randrange(50, 450)
		self.top = self.height - self.PIPE_TOP.get_height()
		self.bottom = self.height + self.GAP

	def move(self):
		self.x -= self.VEL

	def draw(self, win):
		win.blit(self.PIPE_TOP, (self.x, self.top))
		win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

	def collide(self, bird):
		bird_mask = bird.get_mask()
		top_mask = pygame.mask.from_surface(self.PIPE_TOP)
		bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

		top_offset = (self.x - bird.x, self.top - round(bird.y))
		bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

		b_point = bird_mask.overlap(bottom_mask, bottom_offset) #returns None if not overlapped
		t_point = bird_mask.overlap(top_mask, top_offset)

		if t_point or b_point:
			return True

		return False

class Base:
	VEL = 5 #Has to be same as pipe
	WIDTH = BASE_IMG.get_width()
	IMG = BASE_IMG

	def __init__(self, y):
		self.y = y
		self.x1 = 0
		self.x2 = self.WIDTH

	def move(self):
		self.x1 -= self.VEL
		self.x2 -= self.VEL

		if self.x1 + self.WIDTH < 0:
			self.x1 = self.x2 + self.WIDTH

		if self.x2 + self.WIDTH < 0:
			self.x2 = self.x1 + self.WIDTH

	def draw(self, win):
		win.blit(self.IMG, (self.x1, self.y))
		win.blit(self.IMG, (self.x2, self.y))

def draw_window_ai(win, birds, pipes, base, score, gen, birds_alive, genomes, config):
	win.blit(BG_IMG, (0,0))

	for pipe in pipes:
		pipe.draw(win)

	text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
	win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

	text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
	win.blit(text, (10, 10))

	text = STAT_FONT.render("Alive: " + str(birds_alive), 1, (255, 255, 255))
	win.blit(text, (10, 50))

	if button2.update():
		node_names = {0: 'Jump', -1: 'Bottom Pipe Height', -2: 'Top Pipe Height', -3: 'Bird Height'}
		visualize.draw_net(config, genomes[0][1], False, fmt='png', filename='best_neural_net', node_names=node_names) # Save last iteration
		try:
			os.remove('speciation.svg')
			os.remove('avg_fitness.svg')
		except Exception as e:
			pass
		
		menu()

	base.draw(win)

	for bird in birds:
		bird.draw(win)

	win.blit(neural_net_image, (10, WIN_HEIGHT - 70 - neural_net_image.get_height())) # 70 accounts for base
	
	pygame.display.update()

def main_ai(genomes, config):
	global gen
	gen += 1

	node_names = {0: 'Jump', -1: 'Bottom P', -2: 'Top P', -3: 'Bird'}
	visualize.draw_net(config, genomes[0][1], False, fmt='png', filename='best_neural_net', node_names=node_names) # We take the first genome as we can only visualize one neural net
	
	img = Image.open('best_neural_net.png')
	img = img.convert("RGBA")
	datas = img.getdata()

	newData = []
	for item in datas:
	    if item[0] == 255 and item[1] == 255 and item[2] == 255:
	        newData.append((255, 255, 255, 0))
	    else:
	        newData.append(item)

	img.putdata(newData)
	img.save("best_neural_net.png", "PNG")
	
	global neural_net_image
	neural_net_image = pygame.image.load('best_neural_net.png') 

	nets = []
	ge = []
	birds = []
	
	for _, g in genomes:
		net = neat.nn.FeedForwardNetwork.create(g, config)
		nets.append(net)
		birds.append(Bird(230, 350))
		g.fitness = 0
		ge.append(g)

	base = Base(730)
	pipes = [Pipe(600)]
	win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
	clock = pygame.time.Clock()

	score = 0

	run = True
	while run:
		if button2.update():
			menu()

		clock.tick(30)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				node_names = {0: 'Jump', -1: 'Bottom Pipe Height', -2: 'Top Pipe Height', -3: 'Bird Height'}
				visualize.draw_net(config, genomes[0][1], False, fmt='png', filename='best_neural_net', node_names=node_names) # Save last iteration
				try:
					os.remove('speciation.svg')
					os.remove('avg_fitness.svg')
				except Exception as e:
					pass
				run = False
				pygame.quit()
				quit()

		pipe_ind = 0
		if len(birds) > 0:
			if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
				pipe_ind = 1

		else:
			run = False
			break

		for x, bird in enumerate(birds):
			bird.move()
			ge[x].fitness += 0.1

			outputs = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

			if outputs[0] > 0.5:
				bird.jump()

		add_pipe = False
		rem = []
		for pipe in pipes:
			for x, bird in enumerate(birds):
				if pipe.collide(bird):
					ge[x].fitness -= 2
					birds.pop(x)
					nets.pop(x)
					ge.pop(x)

				if not pipe.passed and pipe.x < bird.x:
					pipe.passed = True
					add_pipe = True

			if pipe.x + pipe.PIPE_TOP.get_width() < 0:
				rem.append(pipe)
			
			
			pipe.move()
		
		if add_pipe:
			score += 1

			if score == 50:
				print("Perfect bird out here")

			for g in ge:
				g.fitness += 2
			
			pipes.append(Pipe(600))

		for r in rem:
			pipes.remove(r)

		for x, bird in enumerate(birds):
			if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
				ge[x].fitness -= 2
				birds.pop(x)
				nets.pop(x)
				ge.pop(x)

		base.move()
		
		draw_window_ai(win, birds, pipes, base, score, gen, len(birds), genomes, config)

def draw_window_human(win, bird, pipes, base, score, pregame):
	win.blit(BG_IMG, (0,0))

	for pipe in pipes:
		pipe.draw(win)

	text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
	win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

	base.draw(win)

	bird.draw(win)

	if pregame:
		transparency_size = (500, 800)
		transparency = pygame.Surface(transparency_size)
		transparency.set_alpha(150)
		win.blit(transparency, (0,0))

		text = STAT_FONT_BIG.render("Press Space", 1, (255, 255, 255))
		win.blit(text, (WIN_WIDTH/2- text.get_width()/2, WIN_HEIGHT/2))

		global hs_genopt_popopt
		text = STAT_FONT.render("High Score: " + str(hs_genopt_popopt[0]), 1, (255, 0, 0))
		win.blit(text, (WIN_WIDTH/2- text.get_width()/2, WIN_HEIGHT/2 + 100))

	if button2.update():
		menu()

	pygame.display.update()

def main_human():
	global hs_genopt_popopt

	bird = Bird(230, 350)
	base = Base(730)
	pipes = [Pipe(600)]
	win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
	clock = pygame.time.Clock()

	score = 0

	run_pregame = True
	while run_pregame:
		clock.tick(30)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()
		keys = pygame.key.get_pressed()
		if keys[pygame.K_SPACE]:
			run_pregame = False

		base.move()
		draw_window_human(win, bird, pipes, base, score, True)

	bird.jump() # We just pressed space
	run = True
	while run:
		clock.tick(30)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				if(score > hs_genopt_popopt[0]):
					with open(os.path.join("utils", "hs_genopt_popopt.txt"), "wb") as fp:   #Pickling
						pickle.dump(hs_genopt_popopt, fp)
				run = False
				pygame.quit()
				quit()

		bird.move()
		keys = pygame.key.get_pressed()
		if keys[pygame.K_SPACE]:
			bird.jump()

		add_pipe = False
		rem = []
		for pipe in pipes:
			if pipe.collide(bird):
				if(score > hs_genopt_popopt[0]):
					hs_genopt_popopt[0] = score
				main_human()

			if pipe.x + pipe.PIPE_TOP.get_width() < 0:
				rem.append(pipe)
			
			if not pipe.passed and pipe.x < bird.x:
				pipe.passed = True
				add_pipe = True
			pipe.move()
		
		if add_pipe:
			score += 1

			if score == 50:
				print("Perfect human out here")

			pipes.append(Pipe(600))

		for r in rem:
			pipes.remove(r)

		if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
			if(score > hs_genopt_popopt[0]):
				hs_genopt_popopt[0] = score
				with open(os.path.join("utils", "hs_genopt_popopt.txt"), "wb") as fp:   #Pickling
					pickle.dump(hs_genopt_popopt, fp)
			main_human()

		base.move()
		
		draw_window_human(win, bird, pipes, base, score, False)


def run(config_path):
	config = neat.config.Config(neat.DefaultGenome,
								neat.DefaultReproduction,
								neat.DefaultSpeciesSet,
								neat.DefaultStagnation,
								config_path)

	p = neat.Population(config)

	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	global hs_genopt_popopt
	if hs_genopt_popopt[1] < 1:
		print('Generations set to 1 instead of 0.')
		hs_genopt_popopt[1] = 1

	with open(os.path.join("utils", "hs_genopt_popopt.txt"), "wb") as fp:   #Pickling
		pickle.dump(hs_genopt_popopt, fp)

	global gen
	if gen == hs_genopt_popopt[1]: # If the bird finishes gen gens we reset the number
		gen = 0
	winner = p.run(main_ai, hs_genopt_popopt[1]) #We will run gen generations

	node_names = {0: 'Jump', -1: 'Bottom Pipe Height', -2: 'Top Pipe Height', -3: 'Bird Height'}
	visualize.draw_net(config, winner, False, fmt='png', filename='best_neural_net', node_names=node_names)
	visualize.plot_stats(stats, ylog=False, view=False)
	visualize.plot_species(stats, view=False)

def run_AI():
	global gen
	gen = 0

	global hs_genopt_popopt

	if hs_genopt_popopt[2] < 2:
		print('Population set to 2. P.S: The NN needs at least 2 genomes to function properly.')
		hs_genopt_popopt[2] = 2

	confmodif.conf_file_modify(hs_genopt_popopt[2])

	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, os.path.join("utils", "config-feedforward.txt"))
	run(config_path)

def set_val_gen(value):
    global hs_genopt_popopt
    hs_genopt_popopt[1] = value

def set_val_pop(value):
    global hs_genopt_popopt
    hs_genopt_popopt[2] = value

def menu():
	menu_theme = pygame_menu.themes.THEME_BLUE.copy()
	menu_theme.widget_font = pygame_menu.font.FONT_8BIT

	options = pygame_menu.Menu(800, 500, 'AI Options', onclose=pygame_menu.events.EXIT,
                       theme=menu_theme)

	global hs_genopt_popopt
	valid_chars = ['1','2','3','4','5','6','7','8','9','0']
	options.add_text_input('Generations : ', default=str(hs_genopt_popopt[1]), input_type=pygame_menu.locals.INPUT_INT, valid_chars=valid_chars , maxchar=2, onchange=set_val_gen)
	valid_chars = ['1','2','3','4','5','6','7','8','9','0']
	options.add_text_input('Population : ', default=str(hs_genopt_popopt[2]), input_type=pygame_menu.locals.INPUT_INT, valid_chars=valid_chars ,maxchar=2, onchange=set_val_pop)
	options.add_button('Back', pygame_menu.events.BACK)

	menu = pygame_menu.Menu(800, 500, 'Flappy Bird', theme=menu_theme, onclose=pygame_menu.events.EXIT)
	menu.add_button('AI', run_AI)
	menu.add_button('YOU', main_human)
	menu.add_label('')
	menu.add_label('')
	menu.add_label('')
	menu.add_label('')
	menu.add_button('AI Options', options)
	menu.add_button('Quit', pygame_menu.events.EXIT)
	menu.mainloop(win, fps_limit=30)

if __name__== "__main__":
	menu()