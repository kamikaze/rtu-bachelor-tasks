# RDS subnet

resource "aws_db_subnet_group" "rtu_bachelor_db_subnet_group" {
  count = var.rds_enabled ? 1 : 0
  depends_on = [
    aws_subnet.rtu_bachelor_subnet_eu_north_1a,
    aws_subnet.rtu_bachelor_subnet_eu_north_1b,
    aws_subnet.rtu_bachelor_subnet_eu_north_1c,
  ]
  name = "rtu-bachelor-db-subnet-group"
  subnet_ids = [
    aws_subnet.rtu_bachelor_subnet_eu_north_1a.id,
    aws_subnet.rtu_bachelor_subnet_eu_north_1b.id,
    aws_subnet.rtu_bachelor_subnet_eu_north_1c.id
  ]
}

# RDS

resource "aws_db_instance" "rtu_bachelor_db_master" {
  count = var.rds_enabled ? 1 : 0
  depends_on = [
    aws_security_group.rtu_bachelor_postgres_sg,
    aws_db_subnet_group.rtu_bachelor_db_subnet_group
  ]

  identifier              = "rtu-bachelor-db"
  engine                  = "postgres"
  engine_version          = "15.3"
  db_name                 = "datasets"
  instance_class          = "db.t4g.micro"
  storage_type            = "gp2"
  allocated_storage       = 10
  backup_retention_period = 7
  availability_zone       = "eu-north-1a"
  maintenance_window      = "Sun:04:00-Sun:06:00"
  backup_window           = "02:00-03:00"
  apply_immediately       = true
  db_subnet_group_name    = aws_db_subnet_group.rtu_bachelor_db_subnet_group[count.index].name
  vpc_security_group_ids  = [aws_security_group.rtu_bachelor_postgres_sg.id]
  username                = var.db_username
  password                = var.db_password
  skip_final_snapshot     = true

  tags = {
    Organization = "RTU"
  }
}

locals {
  replicas = [1, 2, 3]
  azs      = ["a", "b", "c"]
}

resource "aws_db_instance" "rtu_bachelor_db_ro_replica" {
  depends_on = [
    aws_security_group.rtu_bachelor_postgres_sg,
    aws_db_subnet_group.rtu_bachelor_db_subnet_group,
    aws_db_instance.rtu_bachelor_db_master
  ]
  count = var.rds_replica_enabled ? length(local.replicas) : 0

  identifier              = "rtu-bachelor-db-ro${local.replicas[count.index]}"
  instance_class          = "db.t4g.micro"
  storage_type            = "gp2"
  backup_retention_period = 7
  availability_zone       = "eu-north-1${local.azs[count.index]}"
  replicate_source_db     = aws_db_instance.rtu_bachelor_db_master[0].identifier
  maintenance_window      = "Sun:04:00-Sun:06:00"
  backup_window           = "02:00-03:00"
  apply_immediately       = true
  vpc_security_group_ids  = [aws_security_group.rtu_bachelor_postgres_sg.id]
  skip_final_snapshot     = true

  tags = {
    Organization = "RTU"
  }
}
